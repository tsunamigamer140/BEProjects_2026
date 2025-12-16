import express from 'express';
import multer from 'multer';
import Transaction from '../models/Transaction.js';
import { classifyTransaction } from '../services/classifier.js';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const pdf = require('pdf-parse');

const router = express.Router();

// ---------------- Multer Config ----------------
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') cb(null, true);
    else cb(new Error('Only PDF files are allowed'), false);
  }
});

// ---------------- Utility Helpers ----------------

function detectPaymentMode(description) {
  const desc = description.toLowerCase();
  if (/upi|phonepe|gpay|paytm|bharatpe/i.test(desc)) return 'UPI';
  if (/atm|withdrawal|cash/i.test(desc)) return 'Cash';
  if (/card|debit|credit|visa|mastercard/i.test(desc)) return 'Card';
  if (/net banking|online transfer|neft|rtgs|imps/i.test(desc)) return 'NetBanking';
  if (/wallet|paytm wallet|phonepe wallet/i.test(desc)) return 'Wallet';
  return 'Other';
}

/**
 * Parse dates like:
 *  - 01-11-2025
 *  - 01/11/25
 *  - 01-Nov-2025
 *  - 01 November 2025
 *  - 2025-11-01
 *  - 2025-Nov-01
 */
function parseDate(dateStr) {
  const monthNames = {
    jan: 0, january: 0,
    feb: 1, february: 1,
    mar: 2, march: 2,
    apr: 3, april: 3,
    may: 4,
    jun: 5, june: 5,
    jul: 6, july: 6,
    aug: 7, august: 7,
    sep: 8, sept: 8, september: 8,
    oct: 9, october: 9,
    nov: 10, november: 10,
    dec: 11, december: 11
  };

  const s = String(dateStr).trim();

  // dd-MMM-yyyy or dd-MMMM-yyyy (01-Nov-2025, 01 November 2025)
  let m = s.match(/(\d{1,2})[-\/\s]([A-Za-z]{3,9})[-\/\s](\d{2,4})/);
  if (m) {
    let day = parseInt(m[1], 10);
    let monKey = m[2].toLowerCase();
    let year = parseInt(m[3], 10);
    if (year < 100) year += 2000;
    const month = monthNames[monKey];
    if (month !== undefined) return new Date(year, month, day);
  }

  // yyyy-MMM-dd or yyyy-MM-dd
  m = s.match(/(\d{2,4})[-\/\s]([A-Za-z]{3,9}|\d{1,2})[-\/\s](\d{1,2})/);
  if (m) {
    let year = parseInt(m[1], 10);
    if (year < 100) year += 2000;
    const mid = m[2];
    let day = parseInt(m[3], 10);
    let month;
    if (/[A-Za-z]/.test(mid)) {
      month = monthNames[mid.toLowerCase()];
    } else {
      month = parseInt(mid, 10) - 1;
    }
    if (month !== undefined) return new Date(year, month, day);
  }

  // Pure numeric: dd-MM-yyyy or yyyy-MM-dd etc.
  const formats = [
    /(\d{1,2})[-\/](\d{1,2})[-\/](\d{2,4})/,
    /(\d{2,4})[-\/](\d{1,2})[-\/](\d{1,2})/
  ];
  for (const format of formats) {
    const match = s.match(format);
    if (match) {
      let day, month, year;
      if (match[1].length === 4) {
        year = parseInt(match[1], 10);
        month = parseInt(match[2], 10) - 1;
        day = parseInt(match[3], 10);
      } else {
        day = parseInt(match[1], 10);
        month = parseInt(match[2], 10) - 1;
        year = parseInt(match[3], 10);
        if (year < 100) year += 2000;
      }
      return new Date(year, month, day);
    }
  }

  return new Date();
}

/**
 * Extract transactions from HDFC-style statement text:
 * 01-Nov-2025 AMAZON PAY*INSTANT PAY -1,299.00 Debit 48,701.00
 *
 * Strategy:
 *  - Look for lines that START with a date
 *  - Capture [date] [description] [amount] [Debit/Credit] [balance]
 *  - Ignore the Summary line (no leading date)
 */
function extractTransactionsFromText(text) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const transactions = [];

  // First, try strict "row" pattern
  const rowRegex = /^(\d{1,2}[-\/][A-Za-z]{3}[-\/]\d{4})\s+(.*?)\s+([+-]?\d[\d,]*(?:\.\d+)?)\s+(Debit|Credit)\s+([\d,]+(?:\.\d+)?)/i;

  for (const line of lines) {
    const m = line.match(rowRegex);
    if (!m) continue;

    const [, dateStr, descRaw, amountStr, typeWord] = m;
    const amount = parseFloat(amountStr.replace(/,/g, ''));
    if (isNaN(amount)) continue;

    const tx = {
      date: parseDate(dateStr),
      description: descRaw.trim(),
      amount: Math.abs(amount),
      type: /credit/i.test(typeWord) ? 'income' : 'expense'
    };

    transactions.push(tx);
  }

  // If we found proper table rows, use only them (skip summary line)
  if (transactions.length) return transactions;

  // ---------- Fallback (generic) ----------
  // Only used for other banks / formats
  const fallback = [];
  const datePattern =
    /(\d{1,2}[-\/](?:\d{1,2}|[A-Za-z]{3,9})[-\/]\d{2,4}|\d{2,4}[-\/](?:\d{1,2}|[A-Za-z]{3,9})[-\/]\d{1,2})/;
  const amountPattern = /(₹|rs\.?\s*)?([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)/i;
  const creditMarkers = /(cr|credit|received|deposit|salary|refund|reversal)/i;
  const debitMarkers = /(dr|debit|payment|purchase|pos|withdrawal|upi|imps|neft|rtgs)/i;

  let current = null;

  for (let line of lines) {
    const dateMatch = line.match(datePattern);

    if (dateMatch) {
      if (current && current.description && current.amount) fallback.push(current);

      current = {
        date: parseDate(dateMatch[1] || dateMatch[0]),
        description: '',
        amount: null,
        type: 'expense'
      };

      const amt = line.match(amountPattern);
      if (amt) {
        const raw = amt[2] || amt[1];
        current.amount = parseFloat(String(raw).replace(/,/g, ''));
        if (current.amount < 0) current.amount = Math.abs(current.amount);
      }

      let desc = line.replace(dateMatch[0], '').replace(amt ? amt[0] : '', '').trim();
      if (desc) current.description = desc;

      if (creditMarkers.test(line)) current.type = 'income';
      if (debitMarkers.test(line)) current.type = 'expense';
    } else if (current) {
      current.description += (current.description ? ' ' : '') + line;

      const amt = line.match(amountPattern);
      if (amt && !current.amount) {
        const raw = amt[2] || amt[1];
        current.amount = parseFloat(String(raw).replace(/,/g, ''));
        if (current.amount < 0) current.amount = Math.abs(current.amount);
      }

      if (creditMarkers.test(line)) current.type = 'income';
      if (debitMarkers.test(line)) current.type = 'expense';
    }
  }

  if (current && current.description && current.amount) fallback.push(current);

  return fallback;
}

async function categorizeTransactions(transactions) {
  const results = [];
  for (const tx of transactions) {
    try {
      const classification = await classifyTransaction(tx.description, tx.amount);
      results.push({
        ...tx,
        category: classification.category,
        confidence: classification.confidence,
        merchant: classification.merchant || '',
        paymentMode: detectPaymentMode(tx.description)
      });
    } catch {
      results.push({
        ...tx,
        category: 'Other',
        confidence: 0.5,
        merchant: '',
        paymentMode: detectPaymentMode(tx.description)
      });
    }
  }
  return results;
}

function generateInsights(transactions) {
  if (!transactions.length) return [];

  const total = transactions.reduce((s, t) => s + t.amount, 0);
  const avg = total / transactions.length;

  const categoryTotals = {};
  transactions.forEach(tx => {
    categoryTotals[tx.category] = (categoryTotals[tx.category] || 0) + tx.amount;
  });

  const topCategory = Object.keys(categoryTotals).length
    ? Object.keys(categoryTotals).reduce((a, b) =>
        categoryTotals[a] > categoryTotals[b] ? a : b
      )
    : null;

  const incomeTx = transactions.filter(t => t.type === 'income');
  const totalIncome = incomeTx.reduce((s, t) => s + t.amount, 0);

  const insights = [
    { icon: '📊', text: `Found ${transactions.length} transactions totaling ₹${total.toLocaleString()}` }
  ];

  if (topCategory) {
    insights.push({
      icon: '🎯',
      text: `Top spending category: ${topCategory} (₹${categoryTotals[topCategory].toLocaleString()})`
    });
  }

  if (avg > 1000) {
    insights.push({
      icon: '💰',
      text: `High average transaction value: ₹${avg.toFixed(0)}` 
    });
  }

  if (incomeTx.length) {
    insights.push({
      icon: '📈',
      text: `Total income recorded: ₹${totalIncome.toLocaleString()}` 
    });
  }

  return insights;
}

async function extractTextFromPDF(buffer) {
  try {
    const data = await pdf(buffer);
    const text = (data.text || '').trim();
    console.log('🧾 Extracted text length:', text.length);
    console.log('🧾 Sample text:', text.slice(0, 400));
    return text;
  } catch (err) {
    console.error('PDF parsing failed:', err);
    return '';
  }
}

// ---------------- Route ----------------

router.post('/upload-pdf', upload.single('pdf'), async (req, res, next) => {
  try {
    console.log('📥 Upload request received');
    console.log('Body fields:', req.body);
    console.log('File metadata:', req.file);

    if (!req.file) {
      return res.status(400).json({ error: 'No PDF uploaded' });
    }

    const userId = req.body.userId || 'guest_user';

    const text = await extractTextFromPDF(req.file.buffer);
    if (!text) {
      return res.status(400).json({ error: 'Could not extract text from PDF' });
    }

    const raw = extractTransactionsFromText(text);
    console.log('🔍 Parsed transactions (raw):', raw.length, raw.slice(0, 3));

    if (!raw.length) {
      return res.status(400).json({ error: 'No transactions found in PDF' });
    }

    const categorized = await categorizeTransactions(raw);

    const saved = [];
    for (const tx of categorized) {
      try {
        const doc = await Transaction.create({
          userId,
          description: tx.description,
          amount: tx.amount,
          type: tx.type,
          paymentMode: tx.paymentMode,
          date: tx.date,
          category: tx.category,
          confidence: tx.confidence,
          merchant: tx.merchant,
          source: 'pdf_upload'
        });
        saved.push(doc);
      } catch (err) {
        console.error('Error saving transaction:', err.message);
      }
    }

    const insights = generateInsights(categorized);
    const totalAmount = categorized.reduce((s, t) => s + t.amount, 0);
    const dates = categorized
      .map(t => new Date(t.date))
      .sort((a, b) => a - b);

    res.json({
      success: true,
      transactions: saved,
      totalAmount,
      dateRange: dates.length
        ? { start: dates[0], end: dates[dates.length - 1] }
        : { start: null, end: null },
      insights,
      extractedCount: categorized.length,
      savedCount: saved.length
    });
  } catch (err) {
    console.error('Upload error:', err);
    next(err);
  }
});

export default router;
