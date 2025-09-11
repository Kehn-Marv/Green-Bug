import React, { useState, useEffect } from 'react';
import { GraduationCap, RefreshCw, Users, CheckCircle, XCircle, Clock, TrendingUp, AlertCircle } from 'lucide-react';
import { getLearningStats, cleanupLearningData, processConsent } from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ResultCard from './ResultCard';

interface LearningStats {
  current_candidates: number;
  pending_consent: number;
  consented_candidates: number;
  denied_candidates: number;
  expired_candidates: number;
  trained_candidates: number;
  consent_rate: number;
  quality_rate: number;
  recent_activity: {
    candidates_last_7_days: number;
    consent_requests_last_7_days: number;
    training_sessions_last_7_days: number;
  };
  quality_distribution?: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  uncertainty_distribution?: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  diversity_distribution?: {
    mean: number;
    std: number;
    min: number;
    max: number;
  };
  safeguards: {
    enabled: boolean;
    violations: any[];
    last_check: string;
  };
}

const LearningSection: React.FC = () => {
  const [learningStats, setLearningStats] = useState<LearningStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [cleanupLoading, setCleanupLoading] = useState(false);

  const fetchLearningStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const stats = await getLearningStats();
      setLearningStats(stats);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to fetch learning statistics');
    } finally {
      setLoading(false);
    }
  };

  const handleCleanup = async () => {
    setCleanupLoading(true);
    try {
      await cleanupLearningData();
      await fetchLearningStats(); // Refresh stats after cleanup
    } catch (err: any) {
      setError(err.message || 'Cleanup failed');
    } finally {
      setCleanupLoading(false);
    }
  };

  useEffect(() => {
    fetchLearningStats();
  }, []);

  const getConsentRateColor = (rate: number) => {
    if (rate >= 0.7) return 'text-success-600';
    if (rate >= 0.4) return 'text-yellow-600';
    return 'text-danger-600';
  };

  const getQualityRateColor = (rate: number) => {
    if (rate >= 0.8) return 'text-success-600';
    if (rate >= 0.6) return 'text-yellow-600';
    return 'text-danger-600';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Self-Learning System</h2>
          <p className="text-gray-600">Monitor learning candidates, consent management, and system training progress</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={handleCleanup}
            disabled={cleanupLoading}
            className="btn-secondary flex items-center space-x-2"
          >
            {cleanupLoading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
            <span>Cleanup</span>
          </button>
          <button
            onClick={fetchLearningStats}
            className="btn-secondary flex items-center space-x-2"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="card border-danger-200 bg-danger-50">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-danger-600" />
            <p className="text-danger-700">{error}</p>
          </div>
        </div>
      )}

      {learningStats && (
        <div className="space-y-6">
          {/* Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="card text-center">
              <Users className="w-8 h-8 text-primary-600 mx-auto mb-3" />
              <p className="text-3xl font-bold text-gray-900 mb-1">
                {learningStats.current_candidates}
              </p>
              <p className="text-sm text-gray-600">Total Candidates</p>
            </div>

            <div className="card text-center">
              <Clock className="w-8 h-8 text-yellow-600 mx-auto mb-3" />
              <p className="text-3xl font-bold text-gray-900 mb-1">
                {learningStats.pending_consent}
              </p>
              <p className="text-sm text-gray-600">Pending Consent</p>
            </div>

            <div className="card text-center">
              <CheckCircle className="w-8 h-8 text-success-600 mx-auto mb-3" />
              <p className="text-3xl font-bold text-gray-900 mb-1">
                {learningStats.consented_candidates}
              </p>
              <p className="text-sm text-gray-600">Consented</p>
            </div>

            <div className="card text-center">
              <GraduationCap className="w-8 h-8 text-indigo-600 mx-auto mb-3" />
              <p className="text-3xl font-bold text-gray-900 mb-1">
                {learningStats.trained_candidates}
              </p>
              <p className="text-sm text-gray-600">Trained</p>
            </div>
          </div>

          {/* Consent and Quality Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ResultCard title="Consent Metrics" icon={Users}>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">Consent Rate</span>
                    <span className={`text-sm font-bold ${getConsentRateColor(learningStats.consent_rate)}`}>
                      {(learningStats.consent_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        learningStats.consent_rate >= 0.7 ? 'bg-success-500' :
                        learningStats.consent_rate >= 0.4 ? 'bg-yellow-500' : 'bg-danger-500'
                      }`}
                      style={{ width: `${learningStats.consent_rate * 100}%` }}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Consented</span>
                    <span className="font-medium text-success-600">{learningStats.consented_candidates}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Denied</span>
                    <span className="font-medium text-danger-600">{learningStats.denied_candidates}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Pending</span>
                    <span className="font-medium text-yellow-600">{learningStats.pending_consent}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-600">Expired</span>
                    <span className="font-medium text-gray-500">{learningStats.expired_candidates}</span>
                  </div>
                </div>
              </div>
            </ResultCard>

            <ResultCard title="Quality Metrics" icon={TrendingUp}>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">Quality Rate</span>
                    <span className={`text-sm font-bold ${getQualityRateColor(learningStats.quality_rate)}`}>
                      {(learningStats.quality_rate * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        learningStats.quality_rate >= 0.8 ? 'bg-success-500' :
                        learningStats.quality_rate >= 0.6 ? 'bg-yellow-500' : 'bg-danger-500'
                      }`}
                      style={{ width: `${learningStats.quality_rate * 100}%` }}
                    />
                  </div>
                </div>

                {learningStats.quality_distribution && (
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Mean Quality</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.quality_distribution.mean.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Std Dev</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.quality_distribution.std.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Min Quality</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.quality_distribution.min.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600">Max Quality</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.quality_distribution.max.toFixed(2)}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </ResultCard>
          </div>

          {/* Recent Activity */}
          <ResultCard title="Recent Activity (Last 7 Days)" icon={Activity}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-primary-600 mb-1">
                  {learningStats.recent_activity.candidates_last_7_days}
                </p>
                <p className="text-sm text-gray-600">New Candidates</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-success-600 mb-1">
                  {learningStats.recent_activity.consent_requests_last_7_days}
                </p>
                <p className="text-sm text-gray-600">Consent Requests</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-indigo-600 mb-1">
                  {learningStats.recent_activity.training_sessions_last_7_days}
                </p>
                <p className="text-sm text-gray-600">Training Sessions</p>
              </div>
            </div>
          </ResultCard>

          {/* Distribution Analysis */}
          {(learningStats.uncertainty_distribution || learningStats.diversity_distribution) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {learningStats.uncertainty_distribution && (
                <ResultCard title="Uncertainty Distribution" icon={BarChart3}>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Mean Uncertainty</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.uncertainty_distribution.mean.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Standard Deviation</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.uncertainty_distribution.std.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Range</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.uncertainty_distribution.min.toFixed(3)} - {learningStats.uncertainty_distribution.max.toFixed(3)}
                      </span>
                    </div>
                  </div>
                </ResultCard>
              )}

              {learningStats.diversity_distribution && (
                <ResultCard title="Diversity Distribution" icon={TrendingUp}>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Mean Diversity</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.diversity_distribution.mean.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Standard Deviation</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.diversity_distribution.std.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Range</span>
                      <span className="font-medium text-gray-900">
                        {learningStats.diversity_distribution.min.toFixed(3)} - {learningStats.diversity_distribution.max.toFixed(3)}
                      </span>
                    </div>
                  </div>
                </ResultCard>
              )}
            </div>
          )}

          {/* Safeguards Status */}
          <ResultCard title="System Safeguards" icon={learningStats.safeguards.enabled ? CheckCircle : AlertCircle}>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Safeguards Enabled</span>
                <div className="flex items-center space-x-2">
                  {learningStats.safeguards.enabled ? (
                    <CheckCircle className="w-4 h-4 text-success-600" />
                  ) : (
                    <XCircle className="w-4 h-4 text-danger-600" />
                  )}
                  <span className={`text-sm font-medium ${learningStats.safeguards.enabled ? 'text-success-600' : 'text-danger-600'}`}>
                    {learningStats.safeguards.enabled ? 'Active' : 'Disabled'}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Violations</span>
                <span className={`text-sm font-medium ${learningStats.safeguards.violations.length === 0 ? 'text-success-600' : 'text-danger-600'}`}>
                  {learningStats.safeguards.violations.length}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Last Check</span>
                <span className="text-sm text-gray-600">
                  {new Date(learningStats.safeguards.last_check).toLocaleString()}
                </span>
              </div>

              {learningStats.safeguards.violations.length > 0 && (
                <div className="mt-4 p-3 bg-danger-50 border border-danger-200 rounded-lg">
                  <p className="text-sm font-medium text-danger-700 mb-2">Recent Violations:</p>
                  <div className="space-y-1">
                    {learningStats.safeguards.violations.slice(-3).map((violation, index) => (
                      <p key={index} className="text-xs text-danger-600">
                        {new Date(violation.timestamp).toLocaleString()}: {violation.failed_checks?.length || 0} failed checks
                      </p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </ResultCard>

          {/* Learning Progress Visualization */}
          <ResultCard title="Learning Progress Overview" icon={GraduationCap}>
            <div className="space-y-6">
              {/* Candidate Pipeline */}
              <div>
                <h4 className="font-medium text-gray-900 mb-4">Candidate Pipeline</h4>
                <div className="flex items-center justify-between">
                  <div className="flex flex-col items-center">
                    <div className="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mb-2">
                      <span className="text-lg font-bold text-primary-600">{learningStats.current_candidates}</span>
                    </div>
                    <span className="text-xs text-gray-600">Total</span>
                  </div>
                  
                  <div className="flex-1 h-1 bg-gray-200 mx-4 rounded-full">
                    <div className="h-1 bg-primary-500 rounded-full" style={{ width: '25%' }} />
                  </div>
                  
                  <div className="flex flex-col items-center">
                    <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center mb-2">
                      <span className="text-lg font-bold text-yellow-600">{learningStats.pending_consent}</span>
                    </div>
                    <span className="text-xs text-gray-600">Pending</span>
                  </div>
                  
                  <div className="flex-1 h-1 bg-gray-200 mx-4 rounded-full">
                    <div className="h-1 bg-yellow-500 rounded-full" style={{ width: '50%' }} />
                  </div>
                  
                  <div className="flex flex-col items-center">
                    <div className="w-12 h-12 bg-success-100 rounded-full flex items-center justify-center mb-2">
                      <span className="text-lg font-bold text-success-600">{learningStats.consented_candidates}</span>
                    </div>
                    <span className="text-xs text-gray-600">Consented</span>
                  </div>
                  
                  <div className="flex-1 h-1 bg-gray-200 mx-4 rounded-full">
                    <div className="h-1 bg-success-500 rounded-full" style={{ width: '75%' }} />
                  </div>
                  
                  <div className="flex flex-col items-center">
                    <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mb-2">
                      <span className="text-lg font-bold text-indigo-600">{learningStats.trained_candidates}</span>
                    </div>
                    <span className="text-xs text-gray-600">Trained</span>
                  </div>
                </div>
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <p className="text-lg font-bold text-gray-900">
                    {(learningStats.consent_rate * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-600">Consent Rate</p>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <p className="text-lg font-bold text-gray-900">
                    {(learningStats.quality_rate * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-600">Quality Rate</p>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <p className="text-lg font-bold text-gray-900">
                    {learningStats.recent_activity.candidates_last_7_days}
                  </p>
                  <p className="text-xs text-gray-600">Recent Candidates</p>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <p className="text-lg font-bold text-gray-900">
                    {learningStats.recent_activity.training_sessions_last_7_days}
                  </p>
                  <p className="text-xs text-gray-600">Recent Training</p>
                </div>
              </div>
            </div>
          </ResultCard>

          {/* Information */}
          <div className="card bg-blue-50 border-blue-200">
            <div className="flex items-start space-x-3">
              <GraduationCap className="w-5 h-5 text-blue-600 mt-0.5" />
              <div>
                <h3 className="font-medium text-blue-900 mb-2">About Self-Learning System</h3>
                <div className="text-sm text-blue-800 space-y-2">
                  <p>
                    The self-learning system continuously improves deepfake detection by learning from high-quality,
                    consented samples. It uses active learning to select the most valuable candidates for training.
                  </p>
                  <p>
                    All learning requires explicit user consent and follows strict safeguards to ensure data privacy
                    and system integrity. The system automatically manages candidate quality, diversity, and consent workflows.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Last Updated */}
          {lastUpdated && (
            <div className="text-center text-sm text-gray-500">
              Learning statistics last updated: {lastUpdated.toLocaleString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LearningSection;