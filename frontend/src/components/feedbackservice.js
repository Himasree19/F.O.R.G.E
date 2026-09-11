// src/services/feedbackService.js

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Sends user feedback and forensic region data to the continuous learning backend.
 */
export const submitForensicFeedback = async ({
  inferenceId,
  originalImageId,
  correctedLabel,
  selectedRegion, // { x, y, width, height } or bounding polygon
  comments,
  confidenceScore,
}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v2/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inference_id: inferenceId,
        original_image_id: originalImageId,
        corrected_label: correctedLabel,
        selected_region: selectedRegion,
        user_comments: comments,
        user_confidence: confidenceScore,
        timestamp: new Date().toISOString(),
      }),
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Failed to send continuous learning feedback:', error);
    throw error;
  }
};