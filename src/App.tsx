import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from './components/ui/toaster';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Verify from './pages/Verify';
import Home from './pages/Home';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  return (
    <Router>
      <Routes>
        {/* Public routes */}
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        
        {/* Protected routes */}
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } 
        />
        
        <Route 
          path="/verify" 
          element={
            <ProtectedRoute>
              <Verify />
            </ProtectedRoute>
          } 
        />
        
        {/* Catch all - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Toaster />
    </Router>
  );
}

export default App;