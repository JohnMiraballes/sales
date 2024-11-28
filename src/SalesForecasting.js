import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, LineElement, PointElement, Title, Tooltip, Legend } from 'chart.js';

// Register the necessary components
ChartJS.register(
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const SalesForecasting = ({ data }) => {
  const [chartData, setChartData] = useState(null);

  const preprocessData = () => {
    if (!data || data.length === 0) {
      console.error('Data is empty or undefined');
      return { inputs: [], outputs: [], productMapping: {} };
    }

    const salesDates = data.map((row) => {
      const date = new Date(row.sales_date);
      return !isNaN(date) ? date.getMonth() + 1 : null; // Extract month as a numeric value
    });

    const products = [...new Set(data.map((row) => row.product_description))];
    const productMapping = Object.fromEntries(products.map((p, i) => [p, i]));

    const quantities = data.map((row) => parseFloat(row.quantity_sold) || 0);

    const inputs = data.map((row, i) => {
      if (salesDates[i] !== null && productMapping[row.product_description] !== undefined) {
        return [salesDates[i], productMapping[row.product_description]];
      }
      return null;
    }).filter((input) => input !== null);

    const outputs = quantities.filter((q, i) => inputs[i] !== null);

    return { inputs, outputs, productMapping };
  };

  const buildModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [2] }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    return model;
  };

  const trainAndPredict = async () => {
    const { inputs, outputs, productMapping } = preprocessData();

    if (inputs.length === 0 || outputs.length === 0) {
      console.error('Invalid input or output data');
      return;
    }

    const xs = tf.tensor2d(inputs, [inputs.length, inputs[0].length]);
    const ys = tf.tensor2d(outputs, [outputs.length, 1]);

    const model = buildModel();
    await model.fit(xs, ys, { epochs: 50 });

    const predictions = [];
    for (let i = 1; i <= 6; i++) {
      Object.keys(productMapping).forEach((product) => {
        const predictionTensor = model.predict(
          tf.tensor2d([[i, productMapping[product]]])
        );
        const predictedValue = predictionTensor.dataSync()[0];
        predictionTensor.dispose();

        predictions.push({
          product,
          sales_date: i,
          predicted: predictedValue,
        });
      });
    }

    visualizeResults(predictions);
  };

  const visualizeResults = (predictions) => {
    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
    const products = [...new Set(predictions.map((p) => p.product))];
    const datasets = products.map((product, index) => ({
      label: product,
      data: predictions
        .filter((p) => p.product === product)
        .map((p) => p.predicted),
      borderColor: colors[index % colors.length],
      backgroundColor: colors[index % colors.length] + '80', // Transparent fill color
      tension: 0.4, // Smooth lines
      pointRadius: 5, // Larger points
      pointHoverRadius: 7, // Points grow on hover
      fill: false,
    }));

    setChartData({
      labels: Array.from({ length: 6 }, (_, i) => `Month ${i + 1}`),
      datasets,
    });
  };

  return (
    <div>
      <button onClick={trainAndPredict}>Train & Predict</button>
      {chartData && (
        <div style={{ width: '800px', height: '500px', margin: 'auto' }}>
          <Line
            data={chartData}
            options={{
              maintainAspectRatio: false,
              responsive: true,
              plugins: {
                legend: {
                  position: 'top',
                  labels: {
                    font: {
                      size: 14,
                    },
                  },
                },
                title: {
                  display: true,
                  text: 'Sales Forecast',
                  font: {
                    size: 18,
                  },
                },
                tooltip: {
                  callbacks: {
                    label: (context) =>
                      `${context.dataset.label}: ${context.raw.toFixed(2)}`,
                  },
                },
              },
              scales: {
                x: {
                  grid: {
                    display: false,
                  },
                },
                y: {
                  grid: {
                    color: '#f3f3f3',
                  },
                },
              },
            }}
          />
        </div>
      )}
    </div>
  );
};

export default SalesForecasting;
