import express from 'express';
import Transaction from '../models/Transaction.js';
import { monthKey } from '../utils/nlp.js';

const router = express.Router();

// GET /api/reports/summary?userId=...&month=YYYY-MM
router.get('/summary', async (req, res, next) => {
  try {
    const { userId, month = monthKey() } = req.query;
    const start = new Date(`${month}-01T00:00:00Z`);
    const end = new Date(start); end.setMonth(end.getMonth() + 1);

    const [byCat, merchants, modes, totals] = await Promise.all([
      Transaction.aggregate([
        { $match: { userId, type: 'expense', date: { $gte: start, $lt: end } } },
        { $group: { _id: '$category', total: { $sum: '$amount' } } },
        { $sort: { total: -1 } }
      ]),
      Transaction.aggregate([
        { $match: { userId, type: 'expense', date: { $gte: start, $lt: end } } },
        { $group: { _id: '$merchant', total: { $sum: '$amount' } } },
        { $sort: { total: -1 } },
        { $limit: 10 }
      ]),
      Transaction.aggregate([
        { $match: { userId, date: { $gte: start, $lt: end } } },
        { $group: { _id: '$paymentMode', total: { $sum: '$amount' } } }
      ]),
      Transaction.aggregate([
        { $match: { userId, date: { $gte: start, $lt: end } } },
        { $group: { _id: '$type', total: { $sum: '$amount' } } }
      ])
    ]);

    const expense = totals.find(t => t._id === 'expense')?.total || 0;
    const income = totals.find(t => t._id === 'income')?.total || 0;

    res.json({ month, expense, income, byCategory: byCat, topMerchants: merchants, byPaymentMode: modes });
  } catch (e) { next(e); }
});

// GET /api/reports/anomalies?userId=...&month=YYYY-MM
router.get('/anomalies', async (req, res, next) => {
  try {
    const { userId, month = monthKey() } = req.query;
    const start = new Date(`${month}-01T00:00:00Z`);
    const end = new Date(start); end.setMonth(end.getMonth() + 1);
    const histStart = new Date(start); histStart.setMonth(histStart.getMonth() - 6);

    const hist = await Transaction.aggregate([
      { $match: { userId, type: 'expense', date: { $gte: histStart, $lt: start } } },
      { $group: { _id: { cat: '$category', m: { $dateToString: { format: '%Y-%m', date: '$date' } } }, total: { $sum: '$amount' } } }
    ]);
    const byCatMonth = {};
    for (const r of hist) {
      const k = r._id.cat;
      byCatMonth[k] = byCatMonth[k] || [];
      byCatMonth[k].push(r.total);
    }

    const cur = await Transaction.aggregate([
      { $match: { userId, type: 'expense', date: { $gte: start, $lt: end } } },
      { $group: { _id: '$category', total: { $sum: '$amount' } } }
    ]);

    const anomalies = [];
    for (const c of cur) {
      const arr = byCatMonth[c._id] || [];
      if (arr.length < 3) continue;
      const avg = arr.reduce((a,b)=>a+b,0) / arr.length;
      const variance = arr.reduce((a,b)=>a + Math.pow(b-avg,2),0) / arr.length;
      const sd = Math.sqrt(variance);
      if (sd > 0 && c.total > avg + 2*sd) {
        anomalies.push({ category: c._id, current: c.total, average: Math.round(avg), threshold: Math.round(avg + 2*sd) });
      }
    }

    res.json({ month, anomalies });
  } catch (e) { next(e); }
});

export default router;


