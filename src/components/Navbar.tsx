import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import {
  Brain,
  Menu,
  FileSearch,
  History,
  Settings,
  Building2,
  BarChart3,
  LogOut,
  User,
  Shield,
  Sparkles,
  Verified,
} from "lucide-react";

export function Navbar() {
  const [user, setUser] = useState(null);
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  const API_BASE_URL = "http://localhost:3000";

  useEffect(() => {
    const getUserFromStorage = () => {
      try {
        const userStr = localStorage.getItem("userInfo");
        if (userStr) {
          const userData = JSON.parse(userStr);
          setUser({
            name: userData.firstname && userData.lastname 
              ? `${userData.firstname} ${userData.lastname}`
              : userData.email,
            email: userData.email,
            ...userData
          });
        }
      } catch (error) {
        console.log('Error parsing stored user data');
      }
    };

    getUserFromStorage();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("authToken");
    localStorage.removeItem("userInfo");
    setUser(null);
    navigate("/login");
  };

  const isActive = (path) => location.pathname === path;

  const navItems = user
    ? [
        {
          href: "/dashboard",
          label: "Dashboard",
          icon: BarChart3,
        },
        {
          href: "/verify",
          label: "Verify Document",
          icon: FileSearch,
        },
      ]
    : [
        { href: "/", label: "Home" },
        { href: "/help", label: "Help" },
      ];

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-white/95 backdrop-blur-xl supports-[backdrop-filter]:bg-white/80 shadow-sm">
      <div className="container flex h-16 items-center justify-between">
        {/* Enhanced Logo */}
        <Link
          to={user ? "/dashboard" : "/"}
          className="flex items-center space-x-3 hover:opacity-90 transition-opacity group"
        >
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-r from-blue-500 to-purple-600 shadow-lg group-hover:shadow-xl transition-shadow">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div className="hidden sm:block">
            <h1 className="font-display text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              SecureX AI
            </h1>
            <p className="text-xs text-muted-foreground -mt-1">Intelligent Verification</p>
          </div>
        </Link>

        {/* Enhanced Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-1">
          {navItems.map((item) => (
            <Link key={item.href} to={item.href}>
              <Button
                variant={isActive(item.href) ? "default" : "ghost"}
                size="sm"
                className={`flex items-center space-x-2 transition-all duration-200 ${
                  isActive(item.href) 
                    ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg" 
                    : "hover:bg-blue-50 hover:text-blue-600"
                }`}
              >
                {item.icon && <item.icon className="h-4 w-4" />}
                <span className="font-medium">{item.label}</span>
              </Button>
            </Link>
          ))}
        </div>

        {/* Enhanced User Actions */}
        <div className="flex items-center space-x-3">
          {user ? (
            <div className="hidden md:flex items-center space-x-4">
              <div className="flex items-center space-x-3 bg-blue-50/80 rounded-lg px-3 py-2 border border-blue-200/50">
                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-green-400 to-blue-500 flex items-center justify-center">
                  <span className="text-white text-sm font-semibold">
                    {user.name?.split(' ').map(n => n[0]).join('') || 'U'}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-medium text-gray-900">
                    {user.name || user.email}
                  </span>
                  <span className="text-xs text-muted-foreground">Verified User</span>
                </div>
              </div>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={handleLogout}
                className="hover:bg-red-50 hover:text-red-600 transition-colors"
              >
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <div className="hidden md:flex items-center space-x-3">
              <Link to="/login">
                <Button variant="ghost" size="sm" className="hover:bg-blue-50 hover:text-blue-600">
                  Login
                </Button>
              </Link>
              <Link to="/verify">
                <Button 
                  size="sm" 
                  className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-lg transition-all duration-200"
                >
                  <Verified className="h-4 w-4 mr-2" />
                  Verify Document
                </Button>
              </Link>
            </div>
          )}

          {/* Enhanced Mobile Menu */}
          <Sheet open={isOpen} onOpenChange={setIsOpen}>
            <SheetTrigger asChild>
              <Button 
                variant="ghost" 
                size="sm" 
                className="md:hidden hover:bg-blue-50 hover:text-blue-600"
              >
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80 border-l border-blue-200/30">
              <div className="flex flex-col space-y-4 mt-8">
                {/* Enhanced Mobile User Info */}
                {user && (
                  <div className="flex items-center space-x-3 p-4 rounded-xl bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200/50">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-r from-green-400 to-blue-500 flex items-center justify-center">
                      <span className="text-white font-semibold">
                        {user.name?.split(' ').map(n => n[0]).join('') || 'U'}
                      </span>
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-gray-900">{user.name || user.email}</p>
                      <p className="text-sm text-muted-foreground">Verified Account</p>
                    </div>
                  </div>
                )}

                {/* Enhanced Mobile Navigation Items */}
                <div className="space-y-2">
                  {navItems.map((item) => (
                    <Link
                      key={item.href}
                      to={item.href}
                      onClick={() => setIsOpen(false)}
                    >
                      <Button
                        variant={isActive(item.href) ? "default" : "ghost"}
                        className={`w-full justify-start space-x-3 transition-all duration-200 ${
                          isActive(item.href)
                            ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg"
                            : "hover:bg-blue-50 hover:text-blue-600"
                        }`}
                      >
                        {item.icon && <item.icon className="h-4 w-4" />}
                        <span className="font-medium">{item.label}</span>
                      </Button>
                    </Link>
                  ))}
                </div>

                {/* Enhanced Mobile Auth Section */}
                {user ? (
                  <div className="pt-4 border-t border-gray-200 space-y-2">
                    <Button
                      variant="ghost"
                      className="w-full justify-start space-x-3 hover:bg-red-50 hover:text-red-600"
                      onClick={() => {
                        handleLogout();
                        setIsOpen(false);
                      }}
                    >
                      <LogOut className="h-4 w-4" />
                      <span>Logout</span>
                    </Button>
                  </div>
                ) : (
                  <div className="pt-4 border-t border-gray-200 space-y-3">
                    <Link to="/login" onClick={() => setIsOpen(false)}>
                      <Button variant="ghost" className="w-full hover:bg-blue-50 hover:text-blue-600">
                        Login to Account
                      </Button>
                    </Link>
                    <Link to="/verify" onClick={() => setIsOpen(false)}>
                      <Button 
                        className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 shadow-lg"
                      >
                        <Verified className="h-4 w-4 mr-2" />
                        Verify Document
                      </Button>
                    </Link>
                  </div>
                )}
              </div>

              {/* Enhanced Mobile Footer */}
              <div className="absolute bottom-4 left-4 right-4">
                <div className="text-center text-xs text-muted-foreground">
                  <p>Powered by SecureX AI</p>
                  <p>Intelligent Academic Verification</p>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </nav>
  );
}
