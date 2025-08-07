document.getElementById('churnForm').addEventListener('submit', async function (e) {
  e.preventDefault();

  const form = e.target;
  const formData = new FormData(form);

  const data = {};
  formData.forEach((value, key) => {
    data[key] = isNaN(value) ? value : Number(value);
  });

  const response = await fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  const probability = result.churn_probability;

  const output = document.getElementById('result');
  output.innerHTML = `Churn Probability: <strong>${(probability * 100).toFixed(2)}%</strong><br>${probability > 0.5 ? '🚨 Customer likely to churn.' : '✅ Customer not likely to churn.'}`;
});
