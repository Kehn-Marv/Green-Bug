import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, AlertCircle, CheckCircle, Download, Eye, Database, FileText, Brain, Shield, Zap } from 'lucide-react';
import { analyzeImage } from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ResultCard from './ResultCard';
import ImageViewer from './ImageViewer';

interface AnalysisResult {
  id: string;
  received_filename: string;
  face: {
    found: boolean;
    confidence: number;
    used_region: [number, number];
  };
  quality: {
    accepted_for_learning: boolean;
    flags: any;
  };
  scores: {
    heuristic_deepfake_score: number;
    deep_model_score?: number;
  };
  features: Record<string, number>;
  attribution_topk: Array<{
    family: string;
    similarity: number;
  }>;
  attribution: {
    predicted_family: string;
    confidence: number;
    ensemble: any;
  };
  robustness: {
    overall_robustness: number;
    stability_metrics: any;
    provenance_stability: any;
  };
  learning_evaluation?: {
    selected: boolean;
    candidate_id?: string;
    scores?: any;
  };
  comprehensive_features: any;
  forensic_reports?: any;
  files: {
    heatmap_url: string;
    overlay_url: string;
  };
  notes: string[];
}

const AnalyzeSection: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [showImageViewer, setShowImageViewer] = useState(false);
  const [viewerImage, setViewerImage] = useState<{ url: string; title: string } | null>(null);
  const [analysisOptions, setAnalysisOptions] = useState({
    stripExif: true,
    enableLearning: true,
    generateReport: false,
    targetLayer: 'layer4.1.conv2'
  });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setResult(null);
      setError(null);
    } else {
      setError('Please select a valid image file');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const analysisResult = await analyzeImage(selectedFile, analysisOptions);
      setResult(analysisResult);
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadImage = async (url: string, filename: string) => {
    try {
      const response = await fetch(`/api${url}`);
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const viewImage = (url: string, title: string) => {
    setViewerImage({ url: `/api${url}`, title });
    setShowImageViewer(true);
  };

  const getScoreColor = (score: number) => {
    if (score < 0.3) return 'text-success-600';
    if (score < 0.7) return 'text-yellow-600';
    return 'text-danger-600';
  };

  const getScoreLabel = (score: number) => {
    if (score < 0.3) return 'Low Risk';
    if (score < 0.7) return 'Medium Risk';
    return 'High Risk';
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Single Image Analysis</h2>
        <p className="text-gray-600">Comprehensive deepfake detection with forensic analysis, attribution, and robustness testing</p>
      </div>

      {/* Analysis Options */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis Options</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={analysisOptions.stripExif}
                onChange={(e) => setAnalysisOptions(prev => ({ ...prev, stripExif: e.target.checked }))}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700">Strip EXIF metadata</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={analysisOptions.enableLearning}
                onChange={(e) => setAnalysisOptions(prev => ({ ...prev, enableLearning: e.target.checked }))}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700">Enable self-learning evaluation</span>
            </label>
          </div>
          <div className="space-y-3">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={analysisOptions.generateReport}
                onChange={(e) => setAnalysisOptions(prev => ({ ...prev, generateReport: e.target.checked }))}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="text-sm text-gray-700">Generate forensic report</span>
            </label>
            <div>
              <label className="block text-sm text-gray-700 mb-1">Target Layer for Grad-CAM</label>
              <select
                value={analysisOptions.targetLayer}
                onChange={(e) => setAnalysisOptions(prev => ({ ...prev, targetLayer: e.target.value }))}
                className="w-full px-3 py-1 border border-gray-300 rounded-md text-sm"
              >
                <option value="layer4.1.conv2">layer4.1.conv2</option>
                <option value="layer3.1.conv2">layer3.1.conv2</option>
                <option value="layer2.1.conv2">layer2.1.conv2</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* File Upload */}
      <div className="card">
        <div
          className={`upload-zone ${dragOver ? 'dragover' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
            className="hidden"
          />
          
          {selectedFile ? (
            <div className="space-y-4">
              <ImageIcon className="w-12 h-12 text-primary-500 mx-auto" />
              <div>
                <p className="font-medium text-gray-900">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <Upload className="w-12 h-12 text-gray-400 mx-auto" />
              <div>
                <p className="text-lg font-medium text-gray-900">Drop your image here</p>
                <p className="text-gray-500">or click to browse files</p>
                <p className="text-sm text-gray-400 mt-2">
                  Supports JPG, PNG, WebP, BMP, TIFF (max 10MB)
                </p>
              </div>
            </div>
          )}
        </div>

        {selectedFile && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="btn-primary flex items-center space-x-2"
            >
              {isAnalyzing ? (
                <>
                  <LoadingSpinner size="sm" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4" />
                  <span>Analyze Image</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="card border-danger-200 bg-danger-50">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-danger-600" />
            <p className="text-danger-700">{error}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Enhanced Main Results */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Detection Scores */}
            <ResultCard title="Detection Scores" icon={AlertCircle}>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">Heuristic Score</span>
                    <span className={`text-sm font-bold ${getScoreColor(result.scores.heuristic_deepfake_score)}`}>
                      {getScoreLabel(result.scores.heuristic_deepfake_score)}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        result.scores.heuristic_deepfake_score < 0.3 ? 'bg-success-500' :
                        result.scores.heuristic_deepfake_score < 0.7 ? 'bg-yellow-500' : 'bg-danger-500'
                      }`}
                      style={{ width: `${result.scores.heuristic_deepfake_score * 100}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {(result.scores.heuristic_deepfake_score * 100).toFixed(1)}% confidence
                  </p>
                </div>

                {result.scores.deep_model_score && (
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700">Deep Model Score</span>
                      <span className={`text-sm font-bold ${getScoreColor(result.scores.deep_model_score)}`}>
                        {getScoreLabel(result.scores.deep_model_score)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          result.scores.deep_model_score < 0.3 ? 'bg-success-500' :
                          result.scores.deep_model_score < 0.7 ? 'bg-yellow-500' : 'bg-danger-500'
                        }`}
                        style={{ width: `${result.scores.deep_model_score * 100}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {(result.scores.deep_model_score * 100).toFixed(1)}% confidence
                    </p>
                  </div>
                )}
              </div>
            </ResultCard>

            {/* Attribution Analysis */}
            <ResultCard title="Attribution Analysis" icon={Brain}>
              <div className="space-y-3">
                {result.attribution && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Predicted Family</span>
                      <span className="text-sm font-medium text-gray-900 capitalize">
                        {result.attribution.predicted_family.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Confidence</span>
                      <span className="text-sm font-medium text-gray-900">
                        {(result.attribution.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-500 h-2 rounded-full"
                        style={{ width: `${result.attribution.confidence * 100}%` }}
                      />
                    </div>
                  </>
                )}
              </div>
            </ResultCard>

            {/* Robustness Analysis */}
            <ResultCard title="Robustness Analysis" icon={Shield}>
              <div className="space-y-3">
                {result.robustness && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Overall Robustness</span>
                      <span className="text-sm font-medium text-gray-900">
                        {(result.robustness.overall_robustness * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-success-500 h-2 rounded-full"
                        style={{ width: `${result.robustness.overall_robustness * 100}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500">
                      Stability against manipulations
                    </p>
                  </>
                )}
              </div>
            </ResultCard>
          </div>

          {/* Main Results */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Face Detection */}
            <ResultCard title="Face Detection" icon={result.face.found ? CheckCircle : AlertCircle}>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Face Found</span>
                  <span className={`text-sm font-medium ${result.face.found ? 'text-success-600' : 'text-gray-500'}`}>
                    {result.face.found ? 'Yes' : 'No'}
                  </span>
                </div>
                {result.face.found && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Confidence</span>
                      <span className="text-sm font-medium text-gray-900">
                        {(result.face.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Region Size</span>
                      <span className="text-sm font-medium text-gray-900">
                        {result.face.used_region[0]} × {result.face.used_region[1]}
                      </span>
                    </div>
                  </>
                )}
              </div>
            </ResultCard>

            {/* Self-Learning Evaluation */}
            {result.learning_evaluation && (
              <ResultCard title="Self-Learning Evaluation" icon={Zap}>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Learning Candidate</span>
                    <span className={`text-sm font-medium ${result.learning_evaluation.selected ? 'text-success-600' : 'text-gray-500'}`}>
                      {result.learning_evaluation.selected ? 'Selected' : 'Not Selected'}
                    </span>
                  </div>
                  {result.learning_evaluation.selected && result.learning_evaluation.candidate_id && (
                    <div className="text-xs text-gray-500">
                      Candidate ID: {result.learning_evaluation.candidate_id}
                    </div>
                  )}
                  {result.learning_evaluation.scores && (
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span>Uncertainty:</span>
                        <span>{(result.learning_evaluation.scores.uncertainty * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Diversity:</span>
                        <span>{(result.learning_evaluation.scores.diversity * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Quality:</span>
                        <span>{(result.learning_evaluation.scores.quality * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  )}
                </div>
              </ResultCard>
            )}
          </div>

          {/* Legacy Attribution Results */}
          {result.attribution_topk && result.attribution_topk.length > 0 && (
            <ResultCard title="Legacy Attribution Analysis" icon={Database}>
              <div className="space-y-3">
                {result.attribution_topk.map((attr, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="text-sm text-gray-700 capitalize">
                      {attr.family.replace(/_/g, ' ')}
                    </span>
                    <div className="flex items-center space-x-2">
                      <div className="w-20 bg-gray-200 rounded-full h-1.5">
                        <div
                          className="bg-primary-500 h-1.5 rounded-full"
                          style={{ width: `${Math.max(attr.similarity * 100, 5)}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900 w-12 text-right">
                        {(attr.similarity * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </ResultCard>
          )}

          {/* Forensic Reports */}
          {result.forensic_reports && (
            <ResultCard title="Forensic Reports" icon={FileText}>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {result.forensic_reports.json && (
                  <div className="text-center">
                    <FileText className="w-8 h-8 text-primary-600 mx-auto mb-2" />
                    <p className="text-sm font-medium text-gray-900">Technical Report (JSON)</p>
                    <button
                      onClick={() => window.open(`/api/files/${result.forensic_reports.json.split('/').pop()}`, '_blank')}
                      className="btn-secondary mt-2 w-full"
                    >
                      Download
                    </button>
                  </div>
                )}
                {result.forensic_reports.html && (
                  <div className="text-center">
                    <FileText className="w-8 h-8 text-success-600 mx-auto mb-2" />
                    <p className="text-sm font-medium text-gray-900">Web Report (HTML)</p>
                    <button
                      onClick={() => window.open(`/api/files/${result.forensic_reports.html.split('/').pop()}`, '_blank')}
                      className="btn-secondary mt-2 w-full"
                    >
                      View
                    </button>
                  </div>
                )}
                {result.forensic_reports.pdf && (
                  <div className="text-center">
                    <FileText className="w-8 h-8 text-danger-600 mx-auto mb-2" />
                    <p className="text-sm font-medium text-gray-900">Forensic Report (PDF)</p>
                    <button
                      onClick={() => window.open(`/api/files/${result.forensic_reports.pdf.split('/').pop()}`, '_blank')}
                      className="btn-secondary mt-2 w-full"
                    >
                      Download
                    </button>
                  </div>
                )}
              </div>
            </ResultCard>
          )}

          {/* Generated Visualizations */}
          <ResultCard title="Generated Visualizations" icon={ImageIcon}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900">Heatmap Analysis</h4>
                <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                  <img
                    src={`/api${result.files.heatmap_url}`}
                    alt="Heatmap"
                    className="w-full h-full object-cover cursor-pointer hover:opacity-90 transition-opacity"
                    onClick={() => viewImage(result.files.heatmap_url, 'Heatmap Analysis')}
                  />
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => viewImage(result.files.heatmap_url, 'Heatmap Analysis')}
                    className="btn-secondary flex-1 flex items-center justify-center space-x-2"
                  >
                    <Eye className="w-4 h-4" />
                    <span>View</span>
                  </button>
                  <button
                    onClick={() => downloadImage(result.files.heatmap_url, `heatmap_${result.id}.png`)}
                    className="btn-secondary flex-1 flex items-center justify-center space-x-2"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download</span>
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="font-medium text-gray-900">Overlay Analysis</h4>
                <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                  <img
                    src={`/api${result.files.overlay_url}`}
                    alt="Overlay"
                    className="w-full h-full object-cover cursor-pointer hover:opacity-90 transition-opacity"
                    onClick={() => viewImage(result.files.overlay_url, 'Overlay Analysis')}
                  />
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => viewImage(result.files.overlay_url, 'Overlay Analysis')}
                    className="btn-secondary flex-1 flex items-center justify-center space-x-2"
                  >
                    <Eye className="w-4 h-4" />
                    <span>View</span>
                  </button>
                  <button
                    onClick={() => downloadImage(result.files.overlay_url, `overlay_${result.id}.png`)}
                    className="btn-secondary flex-1 flex items-center justify-center space-x-2"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download</span>
                  </button>
                </div>
              </div>
            </div>
          </ResultCard>

          {/* Technical Details */}
          <details className="card">
            <summary className="cursor-pointer font-medium text-gray-900 mb-4">
              Comprehensive Technical Details & Features
            </summary>
            <div className="space-y-6">
              {/* Legacy Features */}
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Legacy Heuristic Features</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  {Object.entries(result.features).map(([key, value]) => (
                    <div key={key} className="space-y-1">
                      <span className="text-gray-600 capitalize">
                        {key.replace(/_/g, ' ')}
                      </span>
                      <p className="font-medium text-gray-900">
                        {typeof value === 'number' ? value.toFixed(3) : value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Comprehensive Features */}
              {result.comprehensive_features && (
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Comprehensive Forensic Features</h4>
                  <div className="space-y-4">
                    {Object.entries(result.comprehensive_features).map(([category, features]) => (
                      <div key={category} className="border border-gray-200 rounded-lg p-4">
                        <h5 className="font-medium text-gray-800 mb-2 capitalize">
                          {category.replace(/_/g, ' ')} Features
                        </h5>
                        {typeof features === 'object' && features !== null && (
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                            {Object.entries(features).slice(0, 12).map(([key, value]) => (
                              <div key={key} className="space-y-1">
                                <span className="text-gray-600">
                                  {key.replace(/_/g, ' ')}
                                </span>
                                <p className="font-medium text-gray-900">
                                  {typeof value === 'number' ? value.toFixed(3) : String(value)}
                                </p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Processing Metadata */}
              {result.processing_metadata && (
                <div>
                  <h4 className="font-medium text-gray-900 mb-3">Processing Metadata</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    {Object.entries(result.processing_metadata).map(([key, value]) => (
                      <div key={key} className="space-y-1">
                        <span className="text-gray-600 capitalize">
                          {key.replace(/_/g, ' ')}
                        </span>
                        <p className="font-medium text-gray-900 break-all">
                          {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </details>
        </div>
      )}

      {/* Image Viewer Modal */}
      {showImageViewer && viewerImage && (
        <ImageViewer
          imageUrl={viewerImage.url}
          title={viewerImage.title}
          onClose={() => setShowImageViewer(false)}
        />
      )}
    </div>
  );
};

export default AnalyzeSection;