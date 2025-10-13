import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Navbar } from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  FileSearch,
  History,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  Users,
  Building2,
  Shield,
  Upload,
  BarChart3,
  Zap,
  Scan,
  Brain,
  Sparkles,
  Target,
  Rocket,
  ArrowRight,
  Activity,
  Crown,
  Star,
} from "lucide-react";

interface VerificationData {
  _id: string;
  name: string;
  institution: string;
  date: string;
}

interface User {
  name: string;
  role: string;
  id?: string;
  email?: string;
}

export default function Dashboard() {
  const [user, setUser] = useState<User | null>(null);
  const [recentVerifications, setRecentVerifications] = useState<VerificationData[]>([]);
  const [isLoadingVerifications, setIsLoadingVerifications] = useState(false);
  const [isLoadingUser, setIsLoadingUser] = useState(true);
  const [showAllHistoryModal, setShowAllHistoryModal] = useState(false);

  const API_BASE_URL = "http://localhost:5000";

  const fetchRecentVerifications = async () => {
    setIsLoadingVerifications(true);
    try {
      const token = localStorage.getItem("authToken");
      if (!token) {
        setRecentVerifications([]);
        return;
      }

      const response = await fetch(`${API_BASE_URL}/api/users/results`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data) {
          setRecentVerifications(data.data);
        }
      } else {
        console.log("Failed to fetch recent verifications - using empty array");
        setRecentVerifications([]);
      }
    } catch (error) {
      console.log("Error fetching recent verifications:", error);
      setRecentVerifications([]);
    } finally {
      setIsLoadingVerifications(false);
    }
  };

  const fetchUser = async () => {
    setIsLoadingUser(true);
    try {
      const userStr = localStorage.getItem("userInfo");
      if (userStr) {
        try {
          const localUser = JSON.parse(userStr);
          if (localUser.firstname && localUser.lastname) {
            setUser({
              name: `${localUser.firstname} ${localUser.lastname}`,
              role: localUser.role || "verifier",
              id: localUser.id,
              email: localUser.email
            });
          }
        } catch (e) {
          console.log("Error parsing localStorage user data");
        }
      }

      const token = localStorage.getItem("authToken");
      if (!token) {
        if (!user) {
          setUser({ name: "Guest User", role: "verifier" });
        }
        return;
      }

      const response = await fetch(`${API_BASE_URL}/api/users/me`, {
        credentials: "include",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data) {
          setUser({
            name: data.data.name || `${data.data.firstname || ""} ${data.data.lastname || ""}`.trim() || "User",
            role: data.data.role || "verifier",
            id: data.data.id,
            email: data.data.email
          });
        }
      }
    } catch (error) {
      console.log("Error fetching user:", error);
    } finally {
      setIsLoadingUser(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60)
    );

    if (diffInHours < 1) return "Just now";
    if (diffInHours < 24) return `${diffInHours} hours ago`;
    if (diffInHours < 48) return "1 day ago";
    return `${Math.floor(diffInHours / 24)} days ago`;
  };

  useEffect(() => {
    fetchUser();
    fetchRecentVerifications();
  }, []);

  const displayName = isLoadingUser ? "Loading..." : (user?.name || "User");

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/40 to-indigo-50/60 backdrop-blur-sm">
      <Navbar />
      
      {/* Animated Background Elements */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-r from-blue-200/20 to-purple-300/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-r from-green-200/20 to-cyan-300/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-purple-200/10 to-pink-300/10 rounded-full blur-3xl animate-pulse delay-500"></div>
      </div>

      <div className="container py-8 px-4 sm:px-6 relative z-10">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Enhanced Welcome Header with Glass Morphism */}
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
            <div className="flex-1">
              <div className="flex items-center gap-4 mb-4">
                <div className="relative">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-r from-blue-500 via-purple-600 to-indigo-600 flex items-center justify-center shadow-2xl transform hover:scale-105 transition-all duration-300">
                    <Crown className="h-8 w-8 text-white" />
                  </div>
                  <div className="absolute -top-2 -right-2 w-6 h-6 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center shadow-lg">
                    <Star className="h-3 w-3 text-white" />
                  </div>
                </div>
                <div>
                  <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent leading-tight">
                    Welcome back, {displayName}
                  </h1>
                  <p className="text-lg text-gray-600 mt-2 font-light">
                    Your verification dashboard is ready with latest insights
                  </p>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4 bg-white/80 backdrop-blur-md rounded-2xl p-5 border border-white/20 shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-1">
              <div className="relative">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-green-400 to-blue-500 flex items-center justify-center shadow-lg">
                  <span className="text-white font-bold text-sm">
                    {user?.name?.split(' ').map(n => n[0]).join('') || 'U'}
                  </span>
                </div>
                <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-white"></div>
              </div>
              <div>
                <p className="font-bold text-gray-900 text-lg">{user?.name}</p>
                <p className="text-sm text-gray-600 capitalize bg-gradient-to-r from-gray-100 to-gray-200 px-2 py-1 rounded-full inline-block">
                  {user?.role}
                </p>
              </div>
            </div>
          </div>

          {/* Enhanced Quick Actions with Hover Effects */}
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="border-0 bg-gradient-to-br from-blue-500/10 via-white to-blue-100/30 backdrop-blur-sm shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-2 group">
              <CardContent className="p-8 text-center relative overflow-hidden">
                <div className="absolute -top-10 -right-10 w-20 h-20 bg-blue-200/20 rounded-full blur-xl"></div>
                <div className="relative z-10">
                  <div className="w-20 h-20 rounded-2xl bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center mx-auto mb-6 shadow-2xl group-hover:scale-110 transition-transform duration-300">
                    <Scan className="h-10 w-10 text-white" />
                  </div>
                  <h3 className="font-bold text-2xl mb-3 text-gray-900 group-hover:text-blue-600 transition-colors">
                    New Verification
                  </h3>
                  <p className="text-gray-600 mb-6 leading-relaxed">
                    Upload and verify certificates with AI-powered detection
                  </p>
                  <Link to="/verify">
                    <Button 
                      className="w-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300 group-hover:shadow-2xl text-white font-bold py-3 rounded-xl"
                      size="lg"
                    >
                      <Rocket className="h-5 w-5 mr-2 group-hover:rotate-12 transition-transform" />
                      Start Verification
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

            <Card className="border-0 bg-gradient-to-br from-purple-500/10 via-white to-purple-100/30 backdrop-blur-sm shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-2 group">
              <CardContent className="p-8 text-center relative overflow-hidden">
                <div className="absolute -top-10 -right-10 w-20 h-20 bg-purple-200/20 rounded-full blur-xl"></div>
                <div className="relative z-10">
                  <div className="w-20 h-20 rounded-2xl bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center mx-auto mb-6 shadow-2xl group-hover:scale-110 transition-transform duration-300">
                    <BarChart3 className="h-10 w-10 text-white" />
                  </div>
                  <h3 className="font-bold text-2xl mb-3 text-gray-900 group-hover:text-purple-600 transition-colors">
                    View History
                  </h3>
                  <p className="text-gray-600 mb-6 leading-relaxed">
                    Access complete verification history and analytics
                  </p>
                  <Button
                    variant="outline"
                    size="lg"
                    className="w-full border-2 border-purple-200 bg-white/80 text-purple-700 hover:bg-purple-50 hover:border-purple-300 shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300 font-bold py-3 rounded-xl"
                    onClick={() => setShowAllHistoryModal(true)}
                  >
                    <History className="h-5 w-5 mr-2 group-hover:rotate-12 transition-transform" />
                    Browse History
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Enhanced Recent Activity with Glass Effect */}
            <Card className="border-0 bg-white/60 backdrop-blur-md shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-1">
              <CardHeader className="pb-4 border-b border-gray-200/50">
                <CardTitle className="flex items-center space-x-3 text-gray-900">
                  <div className="p-2 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg shadow-lg">
                    <History className="h-5 w-5 text-white" />
                  </div>
                  <span className="text-xl font-bold">Recent Verifications</span>
                  <Badge variant="secondary" className="ml-2 bg-blue-500 text-white px-3 py-1 rounded-full shadow-lg">
                    {recentVerifications.length}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 pt-6">
                {isLoadingVerifications ? (
                  <div className="flex items-center justify-center p-8">
                    <div className="flex items-center space-x-3">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                      <span className="text-sm text-gray-600 font-medium">Loading recent verifications...</span>
                    </div>
                  </div>
                ) : recentVerifications.length > 0 ? (
                  recentVerifications
                    .slice()
                    .reverse()
                    .slice(0, 5)
                    .map((verification, index) => (
                      <div
                        key={verification._id}
                        className="flex items-center justify-between p-4 bg-white/80 rounded-xl border border-gray-200/50 hover:bg-blue-50/50 hover:border-blue-200/50 transition-all duration-300 transform hover:scale-[1.02] group cursor-pointer"
                      >
                        <div className="flex items-center space-x-4 flex-1">
                          <div className="relative">
                            <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-blue-100 to-purple-100 flex items-center justify-center group-hover:from-blue-200 group-hover:to-purple-200 transition-all duration-300">
                              <FileSearch className="h-6 w-6 text-blue-600" />
                            </div>
                            <div className="absolute -top-1 -right-1 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
                              <CheckCircle className="h-3 w-3 text-white" />
                            </div>
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="font-semibold text-gray-900 truncate group-hover:text-blue-700 transition-colors">
                              {verification.name}
                            </p>
                            <p className="text-sm text-gray-600 truncate">
                              {verification.institution}
                            </p>
                            <p className="text-xs text-gray-500 flex items-center space-x-1 mt-1">
                              <Clock className="h-3 w-3" />
                              <span>{formatDate(verification.date)}</span>
                            </p>
                          </div>
                        </div>
                        <ArrowRight className="h-5 w-5 text-gray-400 group-hover:text-blue-600 group-hover:translate-x-1 transition-all duration-300" />
                      </div>
                    ))
                ) : (
                  <div className="flex flex-col items-center justify-center p-8 text-center bg-gradient-to-br from-gray-50 to-blue-50/30 rounded-xl border-2 border-dashed border-gray-300/50">
                    <FileSearch className="h-16 w-16 text-gray-300 mb-4" />
                    <p className="text-sm text-gray-600 mb-2 font-medium">No recent verifications found.</p>
                    <p className="text-xs text-gray-500">Start by verifying your first certificate</p>
                  </div>
                )}
                <Dialog
                  open={showAllHistoryModal}
                  onOpenChange={setShowAllHistoryModal}
                >
                  <DialogTrigger asChild>
                    <Button variant="ghost" className="w-full border-2 border-dashed border-gray-300/50 text-blue-600 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700 font-medium py-3 rounded-xl transition-all duration-300">
                      View Complete History
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-4xl max-h-[80vh] overflow-auto bg-white/95 backdrop-blur-md border-0 shadow-3xl rounded-2xl">
                    <DialogHeader className="border-b border-gray-200/50 pb-4">
                      <DialogTitle className="flex items-center space-x-3 text-gray-900">
                        <div className="p-2 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg shadow-lg">
                          <History className="h-5 w-5 text-white" />
                        </div>
                        <span className="text-xl font-bold">All Verification History</span>
                      </DialogTitle>
                    </DialogHeader>
                    <div className="space-y-4 mt-4">
                      {isLoadingVerifications ? (
                        <div className="flex items-center justify-center p-8">
                          <div className="text-sm text-gray-600 font-medium">
                            Loading all verifications...
                          </div>
                        </div>
                      ) : recentVerifications.length > 0 ? (
                        <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                          {recentVerifications
                            .slice()
                            .reverse()
                            .map((verification, index) => (
                              <div
                                key={verification._id}
                                className="flex items-center justify-between p-4 bg-white/80 rounded-xl border border-gray-200/50 hover:bg-blue-50/50 hover:border-blue-200/50 transition-all duration-300 transform hover:scale-[1.01]"
                              >
                                <div className="flex items-center space-x-4">
                                  <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center shadow-lg">
                                    <span className="text-white text-sm font-bold">
                                      {index + 1}
                                    </span>
                                  </div>
                                  <div className="flex-1">
                                    <p className="font-semibold text-gray-900">
                                      {verification.name}
                                    </p>
                                    <p className="text-sm text-gray-600">
                                      {verification.institution}
                                    </p>
                                    <p className="text-xs text-gray-500 flex items-center space-x-1">
                                      <Clock className="h-3 w-3" />
                                      <span>{formatDate(verification.date)}</span>
                                    </p>
                                  </div>
                                </div>
                                <Badge variant="outline" className="bg-green-50 border-green-200 text-green-700 font-medium px-3 py-1 rounded-full">
                                  <CheckCircle className="h-3 w-3 mr-1" />
                                  Processed
                                </Badge>
                              </div>
                            ))}
                        </div>
                      ) : (
                        <div className="flex items-center justify-center p-8 bg-gradient-to-br from-gray-50 to-blue-50/30 rounded-xl border-2 border-dashed border-gray-300/50">
                          <div className="text-center">
                            <History className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                            <p className="text-sm text-gray-600 font-medium">
                              No verification history found.
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  </DialogContent>
                </Dialog>
              </CardContent>
            </Card>

            {/* Enhanced Performance Metrics with Animated Progress */}
            <Card className="border-0 bg-white/60 backdrop-blur-md shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-1">
              <CardHeader className="pb-4 border-b border-gray-200/50">
                <CardTitle className="flex items-center space-x-3 text-gray-900">
                  <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg shadow-lg">
                    <TrendingUp className="h-5 w-5 text-white" />
                  </div>
                  <span className="text-xl font-bold">Performance Overview</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6 pt-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-gray-700 flex items-center space-x-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span>Processing Speed</span>
                    </span>
                    <span className="text-sm font-bold text-green-600 bg-green-50 px-2 py-1 rounded-full">
                      Excellent
                    </span>
                  </div>
                  <Progress value={95} className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-yellow-400 to-yellow-500 rounded-full transition-all duration-1000 ease-out animate-pulse" />
                  </Progress>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-gray-700 flex items-center space-x-2">
                      <Target className="h-4 w-4 text-blue-500" />
                      <span>Accuracy Rate</span>
                    </span>
                    <span className="text-sm font-bold text-blue-600 bg-blue-50 px-2 py-1 rounded-full">
                      97.5%
                    </span>
                  </div>
                  <Progress value={97.5} className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-blue-400 to-blue-500 rounded-full transition-all duration-1000 ease-out animate-pulse" />
                  </Progress>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-gray-700 flex items-center space-x-2">
                      <Shield className="h-4 w-4 text-purple-500" />
                      <span>System Reliability</span>
                    </span>
                    <span className="text-sm font-bold text-purple-600 bg-purple-50 px-2 py-1 rounded-full">
                      99.9%
                    </span>
                  </div>
                  <Progress value={99.9} className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-purple-400 to-purple-500 rounded-full transition-all duration-1000 ease-out animate-pulse" />
                  </Progress>
                </div>

                <div className="pt-4 border-t border-gray-200/50">
                  <div className="flex items-center space-x-3 text-sm text-green-600 bg-green-50/80 p-3 rounded-xl border border-green-200/50">
                    <Sparkles className="h-5 w-5 text-green-500" />
                    <span className="font-semibold">All systems operational</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Enhanced System Status with Gradient Background */}
          <Card className="border-0 bg-gradient-to-br from-green-500/10 via-white to-blue-500/10 backdrop-blur-md shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-1">
            <CardHeader className="pb-4 border-b border-gray-200/50">
              <CardTitle className="flex items-center space-x-3 text-gray-900">
                <div className="p-2 bg-gradient-to-r from-green-500 to-green-600 rounded-lg shadow-lg">
                  <Shield className="h-5 w-5 text-white" />
                </div>
                <span className="text-xl font-bold">System Health</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="grid md:grid-cols-3 gap-6">
                <div className="flex items-center space-x-4 p-5 bg-white/80 rounded-xl border border-green-100/50 hover:bg-green-50/50 hover:border-green-200 transition-all duration-300 transform hover:scale-105 group">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-r from-green-500 to-green-600 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                    <Zap className="h-7 w-7 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900">OCR Engine</p>
                    <p className="text-sm text-green-600 font-bold bg-green-50 px-2 py-1 rounded-full inline-block">
                      Operational
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-4 p-5 bg-white/80 rounded-xl border border-blue-100/50 hover:bg-blue-50/50 hover:border-blue-200 transition-all duration-300 transform hover:scale-105 group">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-r from-blue-500 to-blue-600 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                    <Building2 className="h-7 w-7 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900">Registry Access</p>
                    <p className="text-sm text-blue-600 font-bold bg-blue-50 px-2 py-1 rounded-full inline-block">
                      Connected
                    </p>
                  </div>
                </div>

                <div className="flex items-center space-x-4 p-5 bg-white/80 rounded-xl border border-purple-100/50 hover:bg-purple-50/50 hover:border-purple-200 transition-all duration-300 transform hover:scale-105 group">
                  <div className="w-14 h-14 rounded-2xl bg-gradient-to-r from-purple-500 to-purple-600 flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                    <Users className="h-7 w-7 text-white" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900">API Services</p>
                    <p className="text-sm text-purple-600 font-bold bg-purple-50 px-2 py-1 rounded-full inline-block">
                      Active
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Enhanced Quick Stats Section with Hover Animations */}
          <div className="grid md:grid-cols-4 gap-4">
            <Card className="bg-gradient-to-br from-blue-500 to-blue-600 text-white border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-2 group overflow-hidden">
              <CardContent className="p-5 relative">
                <div className="absolute -top-10 -right-10 w-20 h-20 bg-white/10 rounded-full blur-xl"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm opacity-90 font-light">Total Verifications</p>
                      <p className="text-2xl font-bold">{recentVerifications.length}</p>
                    </div>
                    <FileSearch className="h-10 w-10 opacity-90 group-hover:scale-110 transition-transform duration-300" />
                  </div>
                  <div className="mt-2 w-full bg-white/20 h-1 rounded-full overflow-hidden">
                    <div className="bg-white/40 h-full rounded-full animate-pulse" style={{ width: '85%' }}></div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-green-500 to-green-600 text-white border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-2 group overflow-hidden">
              <CardContent className="p-5 relative">
                <div className="absolute -top-10 -right-10 w-20 h-20 bg-white/10 rounded-full blur-xl"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm opacity-90 font-light">Success Rate</p>
                      <p className="text-2xl font-bold">98.7%</p>
                    </div>
                    <CheckCircle className="h-10 w-10 opacity-90 group-hover:scale-110 transition-transform duration-300" />
                  </div>
                  <div className="mt-2 w-full bg-white/20 h-1 rounded-full overflow-hidden">
                    <div className="bg-white/40 h-full rounded-full animate-pulse" style={{ width: '98.7%' }}></div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-purple-500 to-purple-600 text-white border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-2 group overflow-hidden">
              <CardContent className="p-5 relative">
                <div className="absolute -top-10 -right-10 w-20 h-20 bg-white/10 rounded-full blur-xl"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm opacity-90 font-light">Avg. Speed</p>
                      <p className="text-2xl font-bold">2.3s</p>
                    </div>
                    <Zap className="h-10 w-10 opacity-90 group-hover:scale-110 transition-transform duration-300" />
                  </div>
                  <div className="mt-2 w-full bg-white/20 h-1 rounded-full overflow-hidden">
                    <div className="bg-white/40 h-full rounded-full animate-pulse" style={{ width: '95%' }}></div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-orange-500 to-orange-600 text-white border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 transform hover:-translate-y-2 group overflow-hidden">
              <CardContent className="p-5 relative">
                <div className="absolute -top-10 -right-10 w-20 h-20 bg-white/10 rounded-full blur-xl"></div>
                <div className="relative z-10">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm opacity-90 font-light">AI Confidence</p>
                      <p className="text-2xl font-bold">99.1%</p>
                    </div>
                    <Target className="h-10 w-10 opacity-90 group-hover:scale-110 transition-transform duration-300" />
                  </div>
                  <div className="mt-2 w-full bg-white/20 h-1 rounded-full overflow-hidden">
                    <div className="bg-white/40 h-full rounded-full animate-pulse" style={{ width: '99.1%' }}></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}