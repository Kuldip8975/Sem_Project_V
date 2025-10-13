import { Navigate, useLocation } from "react-router-dom";
import { useEffect, useState } from "react";

interface DebugInfo {
    currentPath: string;
    hasToken: boolean;
    hasUserInfo: boolean;
    tokenLength: number;
    userInfoValid: boolean;
    parsedUser: any;
    timestamp: string;
    parseError?: string; // Make this optional
}

export default function ProtectedRoute({ children }: { children: JSX.Element }) {
    const location = useLocation();
    const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
    
    useEffect(() => {
        const token = localStorage.getItem("authToken");
        const userInfo = localStorage.getItem("userInfo");
        
        const debug: DebugInfo = {
            currentPath: location.pathname,
            hasToken: !!token,
            hasUserInfo: !!userInfo,
            tokenLength: token?.length || 0,
            userInfoValid: false,
            parsedUser: null,
            timestamp: new Date().toISOString()
        };
        
        if (userInfo) {
            try {
                debug.parsedUser = JSON.parse(userInfo);
                debug.userInfoValid = true;
            } catch (error) {
                debug.userInfoValid = false;
                debug.parseError = (error as Error).message;
            }
        }
        
        setDebugInfo(debug);
        console.log('🔒 ProtectedRoute Debug:', debug);
        
    }, [location.pathname]);
    
    // Check for auth token and user info in localStorage
    const token = localStorage.getItem("authToken");
    const userInfo = localStorage.getItem("userInfo");
    
    if (!token) {
        console.log('❌ No token found - redirecting to login');
        return <Navigate to="/login" replace />;
    }
    
    if (!userInfo) {
        console.log('❌ No userInfo found - redirecting to login');
        return <Navigate to="/login" replace />;
    }
    
    // Validate userInfo is valid JSON
    try {
        JSON.parse(userInfo);
        console.log('✅ Auth valid - rendering protected component for:', location.pathname);
        return children;
    } catch (error) {
        console.log('❌ Corrupted user data - clearing and redirecting');
        localStorage.removeItem("authToken");
        localStorage.removeItem("userInfo");
        return <Navigate to="/login" replace />;
    }
}