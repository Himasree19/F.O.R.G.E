import React, { useState } from 'react';

export function ContinuousLearningFeedback({ analysisResult, modality }) {
  const [selectedStatus, setSelectedStatus] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  // Normalize modality string to prevent case mismatches
  const currentModality = (modality || 'image').toLowerCase();

  const handleFeedback = async (verifiedLabel) => {
    // 1. Handle Defer / Verify Later option
    if (verifiedLabel === 'DEFER') {
      setSelectedStatus('DEFERRED');
      setToastMessage('Logged to verification queue for later review.');
      return;
    }

    setIsSubmitting(true);

    try {
      let endpoint = `http://127.0.0.1:8000/${currentModality}-feedback`;
      let payload = {};

      // 2. Format payload according to modality
      if (currentModality === 'audio') {
        payload = {
          audio_path: analysisResult.uploaded_file || analysisResult.evidence?.original_filename || '',
          predicted_label: analysisResult.prediction || 'FAKE',
          verified_label: verifiedLabel, // "REAL" or "FAKE"
          confidence: parseFloat(analysisResult.confidence || 0),
        };
      } else if (currentModality === 'text') {
        payload = {
          sentence_id: analysisResult.sentence_id || 1,
          sentence: analysisResult.sentence || analysisResult.text || '',
          predicted_label: analysisResult.prediction || 'FAKE',
          confidence: parseFloat(analysisResult.confidence || 0),
          verified_label: verifiedLabel, // "REAL" or "FAKE"
        };
      } else {
        // Image modality
        payload = {
          image_id: analysisResult.case_id || 'IMG_001',
          verified_label: verifiedLabel, // "AI" or "HUMAN"
        };
      }

      // 3. Post data to FastAPI backend
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (data.success || res.ok) {
        setSelectedStatus(verifiedLabel);
        setToastMessage(`Ground truth recorded: Marked as ${verifiedLabel}.`);
      } else {
        setToastMessage(`Error: ${data.error || 'Failed to record feedback'}`);
      }
    } catch (err) {
      setToastMessage('Network error submitting feedback.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="cl-feedback-wrapper">
      <div className="cl-divider" />
      
      <div className="cl-header">
        <span className="cl-badge-icon">⚡</span>
        <div>
          <h4>HUMAN VERIFICATION & FEEDBACK</h4>
        </div>
      </div>

      {!selectedStatus ? (
        <div className="cl-button-group">
          {currentModality === 'image' ? (
            /* IMAGE BUTTONS: AI vs HUMAN */
            <>
              <button 
                className="cl-btn cl-btn-ai"
                disabled={isSubmitting}
                onClick={() => handleFeedback('AI')}
              >
                <span className="btn-icon">🤖</span> AI
              </button>

              <button 
                className="cl-btn cl-btn-human"
                disabled={isSubmitting}
                onClick={() => handleFeedback('HUMAN')}
              >
                <span className="btn-icon">👤</span> HUMAN
              </button>
            </>
          ) : (
            /* AUDIO & TEXT BUTTONS: REAL vs FAKE */
            <>
              <button 
                className="cl-btn cl-btn-real"
                disabled={isSubmitting}
                onClick={() => handleFeedback('REAL')}
              >
                <span className="btn-icon">✔</span> REAL
              </button>

              <button 
                className="cl-btn cl-btn-fake"
                disabled={isSubmitting}
                onClick={() => handleFeedback('FAKE')}
              >
                <span className="btn-icon">✖</span> FAKE
              </button>
            </>
          )}

          {/* VERIFY LATER (COMMON FOR ALL) */}
          <button 
            className="cl-btn cl-btn-later"
            disabled={isSubmitting}
            onClick={() => handleFeedback('DEFER')}
          >
            <span className="btn-icon">⏱</span> VERIFY LATER
          </button>
        </div>
      ) : (
        <div className={`cl-status-badge status-${selectedStatus.toLowerCase()}`}>
          {selectedStatus === 'DEFERRED' 
            ? '⏱ VERIFICATION DEFERRED' 
            : `🔒 GROUND TRUTH VERIFIED: ${selectedStatus}`}
        </div>
      )}

      {toastMessage && <p className="cl-toast">{toastMessage}</p>}
    </div>
  );
}