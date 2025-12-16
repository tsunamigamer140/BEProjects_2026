import express from 'express';
import Transaction from '../models/Transaction.js';
import { classifyTransaction } from '../services/classifier.js';
import { parseUpiLike } from '../utils/nlp.js';
import fetch from 'node-fetch';
import MLFeedback from '../models/MLFeedback.js';

const router = express.Router();

/**
 * Utility: fallback categorization if classifier fails
 */
function categorizeText(text, defaultCategory = 'Other') {
  if (!text) return defaultCategory;

  if (/zomato|swiggy|food|restaurant/i.test(text)) return 'Food';
  if (/uber|ola|bus|train|cab|metro/i.test(text)) return 'Transport';
  if (/amazon|flipkart|myntra|shopping/i.test(text)) return 'Shopping';
  if (/salary|credited|income|refund/i.test(text)) return 'Income';
  if (/electric|water|gas|bill|recharge/i.test(text)) return 'Utilities';
  if (/movie|entertainment|netflix|spotify/i.test(text)) return 'Entertainment';

  return defaultCategory;
}

// ✅ List transactions
router.get('/', async (req, res, next) => {
  try {
    const { userId } = req.query;
    const items = await Transaction.find({ userId }).sort({ date: -1 }).limit(200);
    res.json(items);
  } catch (e) {
    next(e);
  }
});

// ✅ Add transaction manually
router.post('/', async (req, res, next) => {
  try {
    const { userId, description, amount, type = 'expense', paymentMode = 'UPI', date } = req.body;
    const cls = await classifyTransaction(description, amount);

    // Fallback categorization
    let category = type === 'income' ? 'Income' : cls.category;
    category = categorizeText(description, category);

    const tx = await Transaction.create({
      userId,
      description,
      amount,
      type,
      paymentMode,
      date: date ? new Date(date) : new Date(),
      category,
      confidence: cls.confidence,
      merchant: cls.merchant || '',
    });

    res.json(tx);
  } catch (e) {
    next(e);
  }
});

// ✅ Add transaction from SMS
router.post('/sms', async (req, res, next) => {
  try {
    const { userId, sms } = req.body;
    if (!userId || !sms) return res.status(400).json({ error: 'userId and sms are required' });

    const { amount, merchant } = parseUpiLike(sms);
    const cls = await classifyTransaction(sms, amount);

    // Detect type (income vs expense) - include common spellings/misspellings
    // e.g., "received", "recived", "credited", "salary", "refund", "cashback", "interest", "deposit"
    const incomeRegex = /(salary|credited|credit(ed)?|income|refund|cash\s*back|cashback|interest|deposit|received|recieved|recived|got\s+rs|got\s+₹)/i;
    const type = incomeRegex.test(sms) ? 'income' : 'expense';

    // Fallback categorization
    let category = type === 'income' ? 'Income' : cls.category;
    category = categorizeText(sms, category);

    const tx = await Transaction.create({
      userId,
      description: sms,
      amount: amount || 0,
      type,
      paymentMode: 'UPI',
      merchant,
      category,
      confidence: cls.confidence,
      raw: { sms },
    });

    res.json(tx);
  } catch (e) {
    console.error('SMS route error:', e);
    next(e);
  }
});

// ✅ Update transaction
router.put('/:id', async (req, res, next) => {
  try {
    const { id } = req.params;
    const prev = await Transaction.findById(id);
    const updated = await Transaction.findByIdAndUpdate(id, req.body, { new: true });
    // If category changed, send feedback to ML service
    try {
      if (prev && req.body.category && req.body.category !== prev.category) {
        // Store feedback in DB
        await MLFeedback.create({
          userId: prev.userId,
          text: updated.description,
          label: req.body.category,
          amount: updated.amount,
          merchant: updated.merchant,
          previousCategory: prev.category,
        });
        const ML_URL = process.env.ML_SERVICE_URL || 'http://localhost:5001';
        await fetch(`${ML_URL}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: updated.description,
            amount: updated.amount,
            merchant: updated.merchant,
            label: req.body.category
          }),
          timeout: 1000
        });
      }
    } catch (_) { /* best-effort */ }
    res.json(updated);
  } catch (e) {
    next(e);
  }
});

// ✅ Delete transaction
router.delete('/:id', async (req, res, next) => {
  try {
    const { id } = req.params;
    await Transaction.findByIdAndDelete(id);
    res.json({ ok: true });
  } catch (e) {
    next(e);
  }
});

export default router;
