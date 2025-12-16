
import express from 'express';
import BudgetConfig from '../models/BudgetConfig.js';
import { generateDynamicBudget } from '../services/budget.js';
import { monthKey } from '../utils/nlp.js';

const router = express.Router();

router.get('/', async (req, res, next) => {
  try {
    const { userId, month = monthKey() } = req.query;
    let cfg = await BudgetConfig.findOne({ userId, month });
    res.json(cfg || { userId, month, caps: {} });
  } catch (e) { next(e); }
});

router.post('/generate', async (req, res, next) => {
  try {
    const { userId, income, month = monthKey() } = req.body;
    const caps = await generateDynamicBudget(userId, income, month);
    const cfg = await BudgetConfig.findOneAndUpdate(
      { userId, month }, { userId, month, income, caps }, { upsert: true, new: true }
    );
    res.json(cfg);
  } catch (e) { next(e); }
});

export default router;
