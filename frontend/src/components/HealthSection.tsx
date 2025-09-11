import React, { useState, useEffect } from 'react';
import { Activity, RefreshCw, CheckCircle, AlertCircle, Server, Database, Cpu, HardDrive } from 'lucide-react';
import { getHealth, getDetailedHealth } from '../services/api';
import LoadingSpinner from './LoadingSpinner';
import ResultCard from './ResultCard';

interface HealthData {
  status: string;
  timestamp: string;
  system: {
    memory_usage_percent: number;
    disk_usage_percent: number;
    available_memory_gb: number;
  };
  components: {
    output_directory: boolean;
    fingerprints_db: boolean;
    torch_weights: boolean;
    face_detector: boolean;
  };
  paths: {
    output_dir: string;
    fingerprints: string;
    weights: string;
  };
}

interface DetailedHealthData {
  timestamp: string;
  components: Record<string, string>;
}

const HealthSection: React.FC = () => {
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [detailedHealth, setDetailedHealth] = useState<DetailedHealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchHealth = async () => {
    setLoading(true);
    setError(null);

    try {
      const [health, detailed] = await Promise.all([
        getHealth(),
        getDetailedHealth()
      ]);
      
      setHealthData(health);
      setDetailedHealth(detailed);
      setLastUpdated(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to fetch health data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return 'text-success-600';
      case 'unhealthy':
        return 'text-danger-600';
      default:
        return 'text-yellow-600';
    }
  };

  const getComponentStatus = (available: boolean | string) => {
    if (typeof available === 'boolean') {
      return available ? 'Available' : 'Unavailable';
    }
    return available;
  };

  const getComponentColor = (available: boolean | string) => {
    if (typeof available === 'boolean') {
      return available ? 'text-success-600' : 'text-danger-600';
    }
    
    const status = available.toLowerCase();
    if (status.includes('available') && !status.includes('not')) {
      return 'text-success-600';
    } else if (status.includes('error')) {
      return 'text-danger-600';
    } else {
      return 'text-yellow-600';
    }
  };

  const getUsageColor = (percentage: number) => {
    if (percentage < 70) return 'bg-success-500';
    if (percentage < 85) return 'bg-yellow-500';
    return 'bg-danger-500';
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
          <h2 className="text-2xl font-bold text-gray-900 mb-2">System Health</h2>
          <p className="text-gray-600">Monitor system status and component health</p>
        </div>
        <button
          onClick={fetchHealth}
          className="btn-secondary flex items-center space-x-2"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {error && (
        <div className="card border-danger-200 bg-danger-50">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-danger-600" />
            <p className="text-danger-700">{error}</p>
          </div>
        </div>
      )}

      {healthData && (
        <div className="space-y-6">
          {/* Overall Status */}
          <ResultCard title="Overall System Status" icon={Activity}>
            <div className="text-center">
              <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full ${
                healthData.status === 'healthy' ? 'bg-success-100 text-success-700' : 'bg-danger-100 text-danger-700'
              }`}>
                {healthData.status === 'healthy' ? (
                  <CheckCircle className="w-5 h-5" />
                ) : (
                  <AlertCircle className="w-5 h-5" />
                )}
                <span className="font-semibold capitalize">{healthData.status}</span>
              </div>
              <p className="text-sm text-gray-500 mt-2">
                Last checked: {new Date(healthData.timestamp).toLocaleString()}
              </p>
            </div>
          </ResultCard>

          {/* System Resources */}
          <ResultCard title="System Resources" icon={Server}>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <Cpu className="w-8 h-8 text-primary-600 mx-auto mb-3" />
                <p className="text-sm text-gray-600 mb-2">Memory Usage</p>
                <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
                  <div
                    className={`h-3 rounded-full ${getUsageColor(healthData.system.memory_usage_percent)}`}
                    style={{ width: `${healthData.system.memory_usage_percent}%` }}
                  />
                </div>
                <p className="text-lg font-bold text-gray-900">
                  {healthData.system.memory_usage_percent.toFixed(1)}%
                </p>
                <p className="text-xs text-gray-500">
                  {healthData.system.available_memory_gb.toFixed(1)} GB available
                </p>
              </div>

              <div className="text-center">
                <HardDrive className="w-8 h-8 text-primary-600 mx-auto mb-3" />
                <p className="text-sm text-gray-600 mb-2">Disk Usage</p>
                <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
                  <div
                    className={`h-3 rounded-full ${getUsageColor(healthData.system.disk_usage_percent)}`}
                    style={{ width: `${healthData.system.disk_usage_percent}%` }}
                  />
                </div>
                <p className="text-lg font-bold text-gray-900">
                  {healthData.system.disk_usage_percent.toFixed(1)}%
                </p>
              </div>

              <div className="text-center">
                <Database className="w-8 h-8 text-primary-600 mx-auto mb-3" />
                <p className="text-sm text-gray-600 mb-2">System Status</p>
                <div className={`inline-flex items-center space-x-1 px-3 py-1 rounded-full text-sm font-medium ${
                  healthData.status === 'healthy' ? 'bg-success-100 text-success-700' : 'bg-danger-100 text-danger-700'
                }`}>
                  {healthData.status === 'healthy' ? (
                    <CheckCircle className="w-4 h-4" />
                  ) : (
                    <AlertCircle className="w-4 h-4" />
                  )}
                  <span className="capitalize">{healthData.status}</span>
                </div>
              </div>
            </div>
          </ResultCard>

          {/* Component Status */}
          <ResultCard title="Component Status" icon={CheckCircle}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(healthData.components).map(([component, status]) => (
                <div key={component} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <span className="text-sm font-medium text-gray-700 capitalize">
                    {component.replace(/_/g, ' ')}
                  </span>
                  <div className="flex items-center space-x-2">
                    {status ? (
                      <CheckCircle className="w-4 h-4 text-success-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-danger-600" />
                    )}
                    <span className={`text-sm font-medium ${status ? 'text-success-600' : 'text-danger-600'}`}>
                      {status ? 'Available' : 'Unavailable'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </ResultCard>

          {/* Detailed Component Health */}
          {detailedHealth && (
            <ResultCard title="Detailed Component Health" icon={Activity}>
              <div className="space-y-3">
                {Object.entries(detailedHealth.components).map(([component, status]) => (
                  <div key={component} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                    <span className="text-sm font-medium text-gray-700 capitalize">
                      {component.replace(/_/g, ' ')}
                    </span>
                    <span className={`text-sm font-medium ${getComponentColor(status)}`}>
                      {getComponentStatus(status)}
                    </span>
                  </div>
                ))}
              </div>
            </ResultCard>
          )}

          {/* System Paths */}
          <ResultCard title="System Configuration" icon={Database}>
            <div className="space-y-3">
              {Object.entries(healthData.paths).map(([pathType, path]) => (
                <div key={pathType} className="flex items-start justify-between">
                  <span className="text-sm font-medium text-gray-700 capitalize">
                    {pathType.replace(/_/g, ' ')}
                  </span>
                  <code className="text-xs bg-gray-100 px-2 py-1 rounded font-mono max-w-md truncate">
                    {path}
                  </code>
                </div>
              ))}
            </div>
          </ResultCard>

          {/* Auto-refresh indicator */}
          {lastUpdated && (
            <div className="text-center text-sm text-gray-500">
              Health data auto-refreshes every 30 seconds • Last updated: {lastUpdated.toLocaleString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default HealthSection;