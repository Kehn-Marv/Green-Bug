import React from 'react';
import { Shield, Zap, CheckCircle, AlertCircle, Clock } from 'lucide-react';

interface HeaderProps {
  systemStatus: 'healthy' | 'unhealthy' | 'unknown';
}

const Header: React.FC<HeaderProps> = ({ systemStatus }) => {
  const getStatusIcon = () => {
    switch (systemStatus) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-success-600" />;
      case 'unhealthy':
        return <AlertCircle className="w-4 h-4 text-danger-600" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-600" />;
    }
  };

  const getStatusText = () => {
    switch (systemStatus) {
      case 'healthy':
        return 'System Healthy';
      case 'unhealthy':
        return 'System Issues';
      default:
        return 'Checking Status';
    }
  };

  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Remorph</h1>
              <p className="text-sm text-gray-500">Advanced Deepfake Detection System</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Zap className="w-4 h-4 text-primary-500" />
              <span>CPU-First Forensic Analysis</span>
            </div>
            
            <div className="flex items-center space-x-2 text-sm">
              {getStatusIcon()}
              <span className={`font-medium ${
                systemStatus === 'healthy' ? 'text-success-600' :
                systemStatus === 'unhealthy' ? 'text-danger-600' : 'text-yellow-600'
              }`}>
                {getStatusText()}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;