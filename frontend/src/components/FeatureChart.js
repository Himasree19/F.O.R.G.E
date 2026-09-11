import React from "react";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid
} from "recharts";

function FeatureChart({

  result

}) {

  if (!result) return null;

  const chartData =
    Object.entries(
      result.parameter_contribution
    ).map(([key, value]) => ({

      name: key.toUpperCase(),

      score: value.score
    }));

  return (

    <div className="chart-box">

      <h2>
        Feature Contribution
      </h2>

      <ResponsiveContainer
        width="100%"
        height={350}
      >

        <BarChart data={chartData}>

          <CartesianGrid
            strokeDasharray="3 3"
          />

          <XAxis dataKey="name" />

          <YAxis />

          <Tooltip />

          <Bar dataKey="score" />

        </BarChart>

      </ResponsiveContainer>

    </div>
  );
}

export default FeatureChart;