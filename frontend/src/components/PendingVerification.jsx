// src/pages/PendingVerification.jsx
import React, { useEffect, useState } from "react";
import "./PendingVerification.css"; // We'll add simple styles next

export default function PendingVerification({ onCountUpdate }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [processingId, setProcessingId] = useState(null);

  // Fetch all pending items from backend
  const fetchPendingItems = async () => {
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/pending/list");
      const data = await response.json();
      setItems(data.items || []);
      
      // Update sidebar counter if callback provided
      if (onCountUpdate) {
        onCountUpdate(data.items?.length || 0);
      }
    } catch (error) {
      console.error("Failed to fetch pending items:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPendingItems();
  }, []);

  // Handle final verification (Real / Fake)
  const handleVerify = async (itemId, finalLabel) => {
    setProcessingId(itemId);
    try {
      const response = await fetch("http://localhost:8000/pending/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          item_id: itemId,
          final_label: finalLabel,
        }),
      });

      if (response.ok) {
        // Refresh items list & sidebar count
        await fetchPendingItems();
      } else {
        alert("Failed to verify item. Please try again.");
      }
    } catch (error) {
      console.error("Error submitting verification:", error);
    } finally {
      setProcessingId(null);
    }
  };

  if (loading) {
    return <div className="pending-container">Loading pending verifications...</div>;
  }

  return (
    <div className="pending-container">
      <header className="pending-header">
        <h2>Pending Verification Workspace</h2>
        <p>Review postponed items before sending them to continuous learning.</p>
      </header>

      {items.length === 0 ? (
        <div className="pending-empty">
          <p>🎉 All caught up! No items pending verification.</p>
        </div>
      ) : (
        <div className="pending-list">
          {items.map((item) => (
            <div key={item.id} className={`pending-card modality-${item.modality}`}>
              <div className="card-top">
                <span className={`tag tag-${item.modality}`}>
                  {item.modality.toUpperCase()}
                </span>
                <span className="timestamp">
                  {new Date(item.timestamp).toLocaleString()}
                </span>
              </div>

              <div className="card-content">
                <p className="item-identifier">
                  <strong>Source:</strong> {item.identifier}
                </p>
                <div className="prediction-badge">
                  <span>Prediction: <strong>{item.prediction}</strong></span>
                  <span>Confidence: <strong>{(Number(item.confidence) * 100).toFixed(1)}%</strong></span>
                </div>
              </div>

              <div className="card-actions">
                <span>Verify As:</span>
                <button
                  disabled={processingId === item.id}
                  onClick={() => handleVerify(item.id, "Real")}
                  className="btn-verify btn-real"
                >
                  Real
                </button>
                <button
                  disabled={processingId === item.id}
                  onClick={() => handleVerify(item.id, "Fake")}
                  className="btn-verify btn-fake"
                >
                  Fake
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}