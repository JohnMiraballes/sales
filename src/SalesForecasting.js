import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

// Register the necessary components
ChartJS.register(
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const SalesForecasting = ({ data }) => {
  const [chartData, setChartData] = useState(null);

  const preprocessData = () => {
    console.log('Raw input data:', data);

    if (!data || data.length === 0) {
      console.error('Data is empty or undefined');
      return { inputs: [], outputs: [], productMapping: {} };
    }

    const salesDates = data.map((row) => {
      const date = new Date(row.sales_date);
      return !isNaN(date) ? date.getMonth() + 1 : null;
    });

    const products = [...new Set(data.map((row) => row.product_description))];
    const productMapping = Object.fromEntries(products.map((p, i) => [p, i]));

    const quantities = data.map((row) => parseFloat(row.quantity_sold) || 0);

    const inputs = data
      .map((row, i) => {
        if (
          salesDates[i] !== null &&
          productMapping[row.product_description] !== undefined
        ) {
          return [salesDates[i], productMapping[row.product_description]];
        }
        return null;
      })
      .filter((input) => input !== null);

    const outputs = quantities.filter((q, i) => inputs[i] !== null);

    console.log('Processed inputs:', inputs);
    console.log('Processed outputs:', outputs);
    console.log('Product mapping:', productMapping);

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

    console.log('Training the model...');
    const model = buildModel();
    await model.fit(xs, ys, { epochs: 50 });
    console.log('Model training complete.');

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

    console.log('Predictions:', predictions);

    visualizeResults(predictions);
  };

  const visualizeResults = (predictions) => {
    const colors = ['rgba(255, 99, 132, 0.5)', 'rgba(54, 162, 235, 0.5)', 'rgba(75, 192, 192, 0.5)'];
    const borderColors = ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(75, 192, 192, 1)'];
    const products = [...new Set(predictions.map((p) => p.product))];
    const datasets = products.map((product, index) => ({
      label: product,
      data: predictions
        .filter((p) => p.product === product)
        .map((p) => p.predicted),
      borderColor: borderColors[index % borderColors.length],
      backgroundColor: colors[index % colors.length],
      borderDash: [5, 5], // Dashed lines
      tension: 0.4, // Smooth curves
      fill: true, // Enable gradient fill
      pointRadius: 6, // Larger points
      pointHoverRadius: 8, // Bigger hover effect
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
                      family: "'Roboto', sans-serif",
                    },
                  },
                },
                title: {
                  display: true,
                  text: 'Sales Forecast with Gradient Style',
                  font: {
                    size: 18,
                    family: "'Poppins', sans-serif",
                  },
                },
              },
              scales: {
                x: {
                  grid: {
                    display: true,
                    color: 'rgba(200, 200, 200, 0.5)',
                  },
                },
                y: {
                  grid: {
                    display: true,
                    color: 'rgba(200, 200, 200, 0.5)',
                  },
                  ticks: {
                    callback: (value) => `$${value.toFixed(2)}`, // Add a dollar sign to Y-axis
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
