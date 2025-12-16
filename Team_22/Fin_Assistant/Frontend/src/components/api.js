export const api = {
  token: () => localStorage.getItem('fa_token') || '',
  headers: () => ({ 'Content-Type': 'application/json' }),

  login: async (email, password) => {
    const r = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    if (!r.ok) throw new Error('Login failed');
    return r.json();
  },

  register: async (name, email, password) => {
    const r = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password })
    });
    if (!r.ok) throw new Error('Register failed');
    return r.json();
  },

  listTx: async (userId) => (await fetch(`/api/transactions?userId=${userId}`)).json(),
  addTx: async (payload) => (await fetch('/api/transactions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })).json(),
  delTx: async (id) => (await fetch(`/api/transactions/${id}`, { method: 'DELETE' })).json(),
  smsTx: async (payload) => (await fetch('/api/transactions/sms', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })).json(),
  genBudget: async (userId, income) => (await fetch('/api/budgets/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, income })
  })).json(),
  getBudget: async (userId) => (await fetch(`/api/budgets?userId=${userId}`)).json(),
  chat: async (userId, message) => (await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, message })
  })).json(),

  // ✅ fixed upload
  uploadPDF: async (formData, onProgress) => {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) onProgress(e);
      });

      xhr.addEventListener('load', () => {
        try {
          const response = JSON.parse(xhr.responseText);
          if (xhr.status === 200) {
            resolve(response);
          } else {
            reject(new Error(response.error || response.message || 'Upload failed'));
          }
        } catch (e) {
          reject(new Error('Invalid response from server'));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'));
      });

      xhr.open('POST', '/api/transactions/upload-pdf');
      xhr.send(formData);
    });
  },
};
