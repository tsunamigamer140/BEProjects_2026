
import express from 'express';
import Transaction from '../models/Transaction.js';
import User from '../models/User.js';
import BudgetConfig from '../models/BudgetConfig.js';
import { monthKey } from '../utils/nlp.js';
import { generateDynamicBudget } from '../services/budget.js';

const router = express.Router();

// Very small intent router with regex + keywords
const intents = [
  { name: 'spend_by_category', test: /(spend|spent).*(on|in)\s+([a-z]+)/i },
  { name: 'savings_advice', test: /(save|savings).*(this|current|month)?/i },
  { name: 'safe_daily_spend', test: /(safe|ideal).*(daily|per day).*spend/i },
  { name: 'compare_months', test: /(compare|difference).*(last|previous).*month/i },
  { name: 'budget_status', test: /(budget).*(left|remain|status|cap|limit)|how much.*can.*spend.*left/i },
  { name: 'investment_advice', test: /(invest|investment|portfolio|mutual fund|sip|fd|fixed deposit)/i },
  { name: 'regen_budget', test: /(regenerate|refresh|recompute).*(budget)/i },
];

router.post('/', async (req, res, next) => {
  try {
    const { userId, message } = req.body;
    const text = message || '';
    let reply = "I didn't get that—try asking: “How much did I spend on Food last month?” or “What’s my safe daily spend?”";
    const now = new Date();
    const mk = monthKey(now);

    // 1) Spend by category
    const m1 = text.match(intents[0].test);
    if (m1) {
      const cat = m1[3][0].toUpperCase() + m1[3].slice(1).toLowerCase();
      const start = new Date(mk + '-01T00:00:00Z'); const end = new Date(start); end.setMonth(end.getMonth()+1);
      const agg = await Transaction.aggregate([
        { $match: { userId, category: cat, date: { $gte: start, $lt: end } } },
        { $group: { _id: null, total: { $sum: '$amount' } } }
      ]);
      const total = agg?.[0]?.total || 0;
      reply = `You spent ₹${total.toFixed(0)} on ${cat} this month.`;
      return res.json({ reply });
    }

    // 2) Savings advice
    if (intents[1].test.test(text)) {
      const start = new Date(mk + '-01T00:00:00Z'); const end = new Date(start); end.setMonth(end.getMonth()+1);
      const incomeAgg = await Transaction.aggregate([
        { $match: { userId, type: 'income', date: { $gte: start, $lt: end } } },
        { $group: { _id: null, income: { $sum: '$amount' } } }
      ]);
      const expenseAgg = await Transaction.aggregate([
        { $match: { userId, type: 'expense', date: { $gte: start, $lt: end } } },
        { $group: { _id: null, expense: { $sum: '$amount' } } }
      ]);
      const income = incomeAgg?.[0]?.income || 0;
      const expense = expenseAgg?.[0]?.expense || 0;
      const surplus = income - expense;
      const user = await User.findById(userId).lean();
      const monthlyIncome = user?.monthlyIncome || income;
      const suggested = Math.max(0, Math.round(0.2 * (monthlyIncome || income)));
      reply = `You've spent ₹${expense.toFixed(0)} against income ₹${(monthlyIncome || income).toFixed(0)}; surplus so far is ₹${surplus.toFixed(0)}. Aim to save ~20% (~₹${suggested}). Consider auto‑SIP on salary date.`;
      return res.json({ reply });
    }

    // 3) Safe daily spend
    if (intents[2].test.test(text)) {
      const daysInMonth = new Date(now.getFullYear(), now.getMonth()+1, 0).getDate();
      const remainingDays = daysInMonth - now.getDate() + 1;
      const start = new Date(now.getFullYear(), now.getMonth(), 1);
      const totalAgg = await Transaction.aggregate([
        { $match: { userId, type: 'expense', date: { $gte: start, $lt: now } } },
        { $group: { _id: null, expense: { $sum: '$amount' } } }
      ]);
      const spent = totalAgg?.[0]?.expense || 0;
      const monthlyIncomeAgg = await Transaction.aggregate([
        { $match: { userId, type: 'income', date: { $gte: start, $lt: now } } },
        { $group: { _id: null, income: { $sum: '$amount' } } }
      ]);
      const income = monthlyIncomeAgg?.[0]?.income || 0;

      // If a budget exists, prefer its non-savings cap as spend ceiling
      let targetSpend = 0.7 * income;
      try {
        const user = await User.findById(userId).lean();
        const cfg = await BudgetConfig.findOne({ userId, month: mk }).lean();
        const caps = cfg?.caps || (user?.monthlyIncome ? await generateDynamicBudget(userId, user.monthlyIncome, mk) : null);
        if (caps) {
          const totalCap = Object.entries(caps).reduce((s,[k,v]) => s + (k==='Savings'?0:v||0), 0);
          if (totalCap > 0) targetSpend = totalCap;
        }
      } catch(_) { /* noop */ }
      const remainingBudget = Math.max(0, targetSpend - spent);
      const perDay = remainingDays > 0 ? Math.floor(remainingBudget / remainingDays) : 0;
      reply = `To stay on track, you can spend about ₹${perDay} per day for the rest of the month.`;
      return res.json({ reply });
    }

    // 4) Compare with last month
    if (intents[3].test.test(text)) {
      const startThis = new Date(now.getFullYear(), now.getMonth(), 1);
      const startLast = new Date(now.getFullYear(), now.getMonth()-1, 1);
      const startPrev = new Date(now.getFullYear(), now.getMonth()-2, 1);

      const agg = async (s,e) => (await Transaction.aggregate([
        { $match: { userId, type: 'expense', date: { $gte: s, $lt: e } } },
        { $group: { _id: null, expense: { $sum: '$amount' } } }
      ]))?.[0]?.expense || 0;

      const last = await agg(startLast, startThis);
      const prev = await agg(startPrev, startLast);
      const diff = last - prev;
      reply = `Last month you spent ₹${last.toFixed(0)}. That's ₹${diff >= 0 ? diff.toFixed(0)+' more' : (-diff).toFixed(0)+' less'} than the previous month.`;
      return res.json({ reply });
    }

    // 5) Budget status and remaining
    if (intents[4].test.test(text)) {
      const start = new Date(now.getFullYear(), now.getMonth(), 1);
      const end = new Date(now.getFullYear(), now.getMonth()+1, 1);
      const user = await User.findById(userId).lean();

      // Fetch caps (existing config or dynamic)
      const cfg = await BudgetConfig.findOne({ userId, month: mk }).lean();
      const caps = cfg?.caps || (user?.monthlyIncome ? await generateDynamicBudget(userId, user.monthlyIncome, mk) : {});

      // Current spend by category
      const spendAgg = await Transaction.aggregate([
        { $match: { userId, type: 'expense', date: { $gte: start, $lt: end } } },
        { $group: { _id: '$category', spent: { $sum: '$amount' } } }
      ]);
      const spentByCat = Object.fromEntries(spendAgg.map(a => [a._id || 'Uncategorized', a.spent]));

      // Compute remaining and overspends
      const entries = Object.entries(caps || {});
      const summaries = entries
        .filter(([cat]) => cat !== 'Savings')
        .map(([cat, cap]) => {
          const s = spentByCat[cat] || 0;
          return { cat, cap: Number(cap)||0, spent: s, remain: Math.max(0, (Number(cap)||0) - s) };
        })
        .sort((a,b) => a.remain - b.remain);

      const overs = summaries.filter(x => x.spent > x.cap).slice(0,2).map(x => `${x.cat} (₹${(x.spent - x.cap).toFixed(0)} over)`);
      const tight = summaries.filter(x => x.remain <= 0).length;
      const totalCap = summaries.reduce((s,x)=> s + x.cap, 0);
      const totalSpent = summaries.reduce((s,x)=> s + x.spent, 0);
      const totalRemain = Math.max(0, totalCap - totalSpent);

      const parts = [];
      parts.push(`Overall remaining budget: ₹${totalRemain.toFixed(0)} this month.`);
      if (overs.length) parts.push(`Watch ${overs.join(', ')}.`);
      const savingsCap = Number(caps?.Savings || 0);
      if (savingsCap > 0) {
        const savedSoFarAgg = await Transaction.aggregate([
          { $match: { userId, type: 'income', date: { $gte: start, $lt: end } } },
          { $group: { _id: null, income: { $sum: '$amount' } } }
        ]);
        const incomeMonth = savedSoFarAgg?.[0]?.income || (user?.monthlyIncome || 0);
        const targetSave = Math.min(savingsCap, Math.round(0.2 * (incomeMonth || 0)) || 0);
        parts.push(`Target savings this month: ~₹${targetSave}.`);
      }
      reply = parts.join(' ');
      return res.json({ reply });
    }

    // 6) Investment guidance (very light, rule-of-thumb and goals aware)
    if (intents[5].test.test(text)) {
      const user = await User.findById(userId).lean();
      const income = user?.monthlyIncome || 0;
      const start = new Date(now.getFullYear(), now.getMonth(), 1);
      const end = new Date(now.getFullYear(), now.getMonth()+1, 1);
      const expenseAgg = await Transaction.aggregate([
        { $match: { userId, type: 'expense', date: { $gte: start, $lt: end } } },
        { $group: { _id: null, expense: { $sum: '$amount' } } }
      ]);
      const expense = expenseAgg?.[0]?.expense || 0;
      const possibleSip = Math.max(0, Math.round(0.15 * (income || (expense*1.2)))) || 0;

      reply = `Consider building an emergency fund of 3–6 months expenses (₹${(Math.max(expense, income*0.6)*3).toFixed(0)}–₹${(Math.max(expense, income*0.6)*6).toFixed(0)}). A simple allocation could be 70% equity index funds, 20% short‑duration debt, 10% gold; reduce equity if you prefer lower risk. Start a monthly SIP around ₹${possibleSip} and review yearly.`;
      return res.json({ reply });
    }

    // 7) Regenerate budget now (tool action)
    if (intents[6].test.test(text)) {
      const user = await User.findById(userId).lean();
      const income = user?.monthlyIncome || 0;
      if (!income) {
        return res.json({ reply: 'I need your monthly income to regenerate your budget. Please set it in settings.' });
      }
      const caps = await generateDynamicBudget(userId, income, mk);
      await BudgetConfig.findOneAndUpdate(
        { userId, month: mk },
        { userId, month: mk, income, caps },
        { upsert: true, new: true }
      );
      return res.json({ reply: 'Done. I regenerated your budget for this month based on your recent spending.' });
    }

    // LLM fallback for dynamic Q&A if no regex intent matched
    try {
      const PROVIDER = (process.env.LLM_PROVIDER || '').toLowerCase();

      // Build lightweight financial context
      const startThis = new Date(now.getFullYear(), now.getMonth(), 1);
      const startLast = new Date(now.getFullYear(), now.getMonth()-1, 1);
      const startPrev = new Date(now.getFullYear(), now.getMonth()-2, 1);

      const sumAgg = async (match) => (await Transaction.aggregate([
        { $match: match },
        { $group: { _id: null, total: { $sum: '$amount' } } }
      ]))?.[0]?.total || 0;

      const expenseMTD = await sumAgg({ userId, type: 'expense', date: { $gte: startThis, $lt: now } });
      const incomeMTD = await sumAgg({ userId, type: 'income', date: { $gte: startThis, $lt: now } });
      const lastMonthExpense = await sumAgg({ userId, type: 'expense', date: { $gte: startLast, $lt: startThis } });
      const prevMonthExpense = await sumAgg({ userId, type: 'expense', date: { $gte: startPrev, $lt: startLast } });

      const user = await User.findById(userId).lean();
      const cfg = await BudgetConfig.findOne({ userId, month: mk }).lean();
      const caps = cfg?.caps || (user?.monthlyIncome ? await generateDynamicBudget(userId, user.monthlyIncome, mk) : {});

      // Compose prompt
      const sys = 'You are a precise, cautious personal finance assistant for an Indian user. Always cite rupee amounts with the 9 sign. Provide short, actionable answers with one concrete next step. Avoid speculation and never fabricate data.';
      const ctx = {
        month: mk,
        monthlyIncome: user?.monthlyIncome || 0,
        expenseMonthToDate: Math.round(expenseMTD),
        incomeMonthToDate: Math.round(incomeMTD),
        lastMonthExpense: Math.round(lastMonthExpense),
        prevMonthExpense: Math.round(prevMonthExpense),
        budgetCaps: caps
      };
      const userMsg = `User message: ${text}`;
      const contextMsg = `Context JSON: ${JSON.stringify(ctx).slice(0, 5000)}`;

      // Provider: Ollama (FREE local LLM)
      if (PROVIDER === 'ollama') {
        const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';
        const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'llama3.1:8b';
        const prompt = `${sys}\n\n${contextMsg}\n\n${userMsg}`;
        const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: OLLAMA_MODEL, prompt, stream: false, options: { temperature: 0.2, num_predict: 250 } }),
        });
        if (resp.ok) {
          const data = await resp.json();
          const textOut = (data?.response || '').trim();
          if (textOut) return res.json({ reply: textOut });
        }
        return res.json({ reply });
      }

      // Default: OpenAI (requires API key)
      const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
      if (OPENAI_API_KEY) {
        const resp = await fetch('https://api.openai.com/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${OPENAI_API_KEY}` },
          body: JSON.stringify({
            model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
            messages: [
              { role: 'system', content: sys },
              { role: 'user', content: userMsg },
              { role: 'user', content: contextMsg }
            ],
            temperature: 0.2,
            max_tokens: 250
          })
        });
        if (resp.ok) {
          const data = await resp.json();
          const textOut = data?.choices?.[0]?.message?.content?.trim();
          if (textOut) return res.json({ reply: textOut });
        }
      }
      return res.json({ reply });
    } catch (_) {
      return res.json({ reply });
    }
  } catch (e) { next(e); }
});

export default router;
