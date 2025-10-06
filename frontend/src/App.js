import React, { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setPrediction("");
    setConfidence(null);
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log("Prediction result:", data);

      if (data.class_name) {
        setPrediction(data.class_name);
        setConfidence(data.confidence);
      } else if (data.error) {
        setPrediction("Error: " + data.error);
      } else {
        setPrediction("No result");
      }
    } catch (error) {
      console.error("Error:", error);
      setPrediction("Error predicting disease");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>🌿 Plant Disease Detection</h1>

      <div className="upload-section">
        <input type="file" accept="image/*" onChange={handleImageChange} />

        {preview && (
          <div className="image-container">
            <img src={preview} alt="Preview" />
          </div>
        )}

        <button onClick={handlePredict} disabled={loading}>
          {loading ? "Predicting..." : "Predict Disease"}
        </button>

        {prediction && (
          <div className="result-card">
            <h2>🩺 Result:</h2>
            <p>
              <strong>{prediction}</strong>
              {confidence !== null && (
                <span style={{ color: "#1b4332" }}>
                  {" "}
                  ({confidence.toFixed(2)}%)
                </span>
              )}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

