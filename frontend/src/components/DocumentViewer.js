import React, {
    useEffect,
    useRef,
    useState
  } from "react";
  
  import {
    Document,
    Page,
    pdfjs
  } from "react-pdf";
  
  import { renderAsync } from "docx-preview";
  
  import "react-pdf/dist/esm/Page/TextLayer.css";
  import "react-pdf/dist/esm/Page/AnnotationLayer.css";
  
  import "./DocumentViewer.css";
  
  // ==========================================
  // PDF WORKER FIX
  // ==========================================
  
  pdfjs.GlobalWorkerOptions.workerSrc =
    `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;
  
  function DocumentViewer({
  
    uploadedFile,
  
    result
  
  }) {
  
    const viewerRef = useRef(null);
  
    const [numPages, setNumPages] =
      useState(null);
  
    const [docxLoaded, setDocxLoaded] =
      useState(false);
  
    // ==========================================
    // DOCX RENDERING
    // ==========================================
  
    useEffect(() => {
  
      if (
        uploadedFile &&
        uploadedFile.name.endsWith(".docx")
      ) {
  
        const reader = new FileReader();
  
        reader.onload = async (event) => {
  
          const arrayBuffer =
            event.target.result;
  
          if (viewerRef.current) {
  
            viewerRef.current.innerHTML = "";
  
            await renderAsync(
  
              arrayBuffer,
  
              viewerRef.current
  
            );
  
            setDocxLoaded(true);
          }
        };
  
        reader.readAsArrayBuffer(
          uploadedFile
        );
      }
  
    }, [uploadedFile]);
  
    // ==========================================
    // PDF LOAD SUCCESS
    // ==========================================
  
    const onDocumentLoadSuccess = ({
      numPages
    }) => {
  
      setNumPages(numPages);
    };
  
    // ==========================================
    // EMPTY
    // ==========================================
  
    if (!uploadedFile) {
  
      return null;
    }
  
    return (
  
      <div className="viewer-container">
  
        <h2>
          Live Document Viewer
        </h2>
  
        {/* ======================================
            PDF VIEWER
        ====================================== */}
  
        {uploadedFile.name.endsWith(".pdf") && (
  
          <div className="pdf-container">
  
            <Document
  
              file={uploadedFile}
  
              onLoadSuccess={
                onDocumentLoadSuccess
              }
  
              loading={
                <p>
                  Loading PDF...
                </p>
              }
  
            >
  
              {Array.from(
  
                new Array(numPages),
  
                (el, index) => (
  
                  <Page
  
                    key={`page_${index + 1}`}
  
                    pageNumber={index + 1}
  
                    width={800}
  
                  />
                )
              )}
  
            </Document>
  
          </div>
        )}
  
        {/* ======================================
            DOCX VIEWER
        ====================================== */}
  
        {uploadedFile &&
          uploadedFile.name.endsWith(".docx") && (
  
          <div className="docx-container">
  
            {!docxLoaded && (
  
              <p>
                Rendering DOCX...
              </p>
            )}
  
            <div ref={viewerRef}></div>
  
          </div>
        )}
  
        {/* ======================================
            IMAGE PREVIEW
        ====================================== */}
  
        {uploadedFile &&
          (
            uploadedFile.name.endsWith(".png") ||
            uploadedFile.name.endsWith(".jpg") ||
            uploadedFile.name.endsWith(".jpeg")
          ) && (
  
          <div className="image-preview">
  
            <img
  
              src={URL.createObjectURL(
                uploadedFile
              )}
  
              alt="Uploaded Preview"
  
              className="preview-image"
            />
  
          </div>
        )}
  
        {/* ======================================
            INLINE AI HEATMAP
        ====================================== */}
  
        {result &&
          result.full_document && (
  
          <div className="inline-analysis">
  
            <h3>
              Inline AI Heatmap
            </h3>
  
            <div className="heatmap-text">
  
              {result.full_document.map(
                (item, index) => (
  
                  <span
  
                    key={index}
  
                    className={`heatmap-${item.color}`}
  
                    title={item.reason}
  
                  >
  
                    {item.sentence}{" "}
  
                  </span>
                )
              )}
  
            </div>
  
          </div>
        )}
  
      </div>
    );
  }
  
  export default DocumentViewer;