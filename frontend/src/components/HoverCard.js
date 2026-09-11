import React from "react";

function HoverCard({

  sentence

}) {

  if (!sentence) return null;

  return (

    <div className="hover-card">

      <h3>
        Sentence Analysis
      </h3>

      <p>

        <strong>AI Score:</strong>

        {" "}
        {sentence.score}%

      </p>

      <p>

        <strong>Risk:</strong>

        {" "}
        {sentence.risk}

      </p>

      <p>

        <strong>Reason:</strong>

        {" "}
        {sentence.reason}

      </p>

    </div>
  );
}

export default HoverCard;