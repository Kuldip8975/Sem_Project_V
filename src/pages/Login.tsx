import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Shield,
  Eye,
  EyeOff,
  CheckCircle,
  AlertCircle,
  ArrowLeft,
  Mail,
  Lock,
  Brain,
  GraduationCap,
  Verified,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import graduationImage from "@/assets/graduation-digital.jpg";

export default function Login() {
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [role, setRole] = useState("verifier");
  const [passwordStrength, setPasswordStrength] = useState(0);
  const [otp, setOtp] = useState("");
  const [otpSent, setOtpSent] = useState(false);
  const [activeTab, setActiveTab] = useState("login");
  const [registeredEmail, setRegisteredEmail] = useState(""); // New state to store registered email
  const { toast } = useToast();
  const navigate = useNavigate();

  // Use consistent API base URL - consider using environment variables
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

  // Effect to set email when switching to login tab after registration
  useEffect(() => {
    if (activeTab === "login" && registeredEmail) {
      setEmail(registeredEmail);
    }
  }, [activeTab, registeredEmail]);

  const calculatePasswordStrength = (pass) => {
    let strength = 0;
    if (pass.length >= 8) strength += 25;
    if (/[A-Z]/.test(pass)) strength += 25;
    if (/[0-9]/.test(pass)) strength += 25;
    if (/[^A-Za-z0-9]/.test(pass)) strength += 25;
    return strength;
  };

  const getPasswordStrengthColor = (strength) => {
    if (strength >= 75) return "text-green-600";
    if (strength >= 50) return "text-yellow-600";
    return "text-red-600";
  };

  const getPasswordStrengthBgColor = (strength) => {
    if (strength >= 75) return "bg-green-500";
    if (strength >= 50) return "bg-yellow-500";
    return "bg-red-500";
  };

  const handlePasswordChange = (value) => {
    setPassword(value);
    setPasswordStrength(calculatePasswordStrength(value));
  };

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleSendOtp = async (e) => {
    e.preventDefault();
    
    if (!email) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please enter your email address",
      });
      return;
    }

    if (!validateEmail(email)) {
      toast({
        variant: "destructive",
        title: "Invalid Email",
        description: "Please enter a valid email address",
      });
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/users/send-otp`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setOtpSent(true);
        
        // For testing - show the OTP in a toast (remove in production!)
        if (data.testOtp) {
          toast({
            title: "OTP Generated (Test Mode)",
            description: `Your test OTP is: ${data.testOtp}. In production, this would be sent via email.`,
            duration: 10000,
          });
        } else {
          toast({
            title: "OTP Sent Successfully",
            description: "Check your email for the verification code",
          });
        }
      } else {
        throw new Error(data.message || "Failed to send OTP");
      }
    } catch (error) {
      console.error("Send OTP error:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: error.message || "Failed to send OTP. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerifyOtp = async (e) => {
    e.preventDefault();
    
    if (!otp || otp.length !== 6) {
      toast({
        variant: "destructive",
        title: "Invalid OTP",
        description: "Please enter a valid 6-digit OTP",
      });
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/users/verify-otp`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, otp }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        toast({
          title: "Welcome to SecureX AI!",
          description: "Authentication successful! Redirecting to your dashboard...",
        });

        // Store token and user info
        if (data.token) {
          localStorage.setItem("authToken", data.token);
        }
        if (data.user) {
          localStorage.setItem("userInfo", JSON.stringify(data.user));
        }

        // Navigate to dashboard
        setTimeout(() => navigate("/dashboard"), 1500);
      } else {
        throw new Error(data.message || "Invalid OTP");
      }
    } catch (error) {
      console.error("Verify OTP error:", error);
      toast({
        variant: "destructive",
        title: "Verification Failed",
        description: error.message || "Invalid OTP. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    
    // Enhanced validation
    if (!firstName || !lastName || !email || !password) {
      toast({
        variant: "destructive",
        title: "Missing Information",
        description: "All fields are required",
      });
      return;
    }

    if (firstName.length < 2 || lastName.length < 2) {
      toast({
        variant: "destructive",
        title: "Invalid Name",
        description: "Name must be at least 2 characters long",
      });
      return;
    }

    if (!validateEmail(email)) {
      toast({
        variant: "destructive",
        title: "Invalid Email",
        description: "Please enter a valid email address",
      });
      return;
    }

    if (password.length < 8) {
      toast({
        variant: "destructive",
        title: "Weak Password",
        description: "Password must be at least 8 characters long",
      });
      return;
    }

    if (passwordStrength < 50) {
      toast({
        variant: "destructive",
        title: "Weak Password",
        description: "Please choose a stronger password",
      });
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/users/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          firstname: firstName,
          lastname: lastName,
          email,
          password,
          role: "user" // Default role
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Store the registered email before resetting the form
        setRegisteredEmail(email);
        
        toast({
          title: "Welcome to SecureX AI!",
          description: "Your account has been created successfully! You can now login.",
        });

        // Reset form
        setFirstName("");
        setLastName("");
        setPassword("");
        setPasswordStrength(0);
        // Don't reset email here - we want to keep it for the login tab

        // Switch to login tab
        setActiveTab("login");
      } else {
        throw new Error(data.message || "Registration failed");
      }
    } catch (error) {
      console.error("Registration error:", error);
      toast({
        variant: "destructive",
        title: "Registration Failed",
        description: error.message || "Something went wrong. Please try again.",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const resetOtpFlow = () => {
    setOtpSent(false);
    setOtp("");
  };

  const handleKeyPress = (e) => {
    // Allow only numbers for OTP input
    if (otpSent && !/^\d$/.test(e.key)) {
      e.preventDefault();
    }
  };

  // Function to clear registered email when manually switching tabs
  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === "register") {
      setRegisteredEmail(""); // Clear the stored email when switching to register tab
    }
  };

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Left Panel - Enhanced Image Section */}
      <div className="hidden lg:flex lg:w-1/2 relative">
        <img
          src={graduationImage}
          alt="SakshAI Academic Verification"
          className="absolute inset-0 w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/80 to-purple-900/80" />

        {/* Enhanced Overlay Content */}
        <div className="relative z-10 flex flex-col justify-center p-12 text-white">
          <div className="max-w-md">
            <div className="flex items-center space-x-3 mb-8">
              <div className="flex items-center justify-center w-12 h-12 rounded-lg bg-white/20 backdrop-blur-sm">
                <Brain className="h-7 w-7 text-blue-300" />
              </div>
              <div>
                <h1 className="font-display text-2xl font-bold bg-gradient-to-r from-blue-300 to-purple-300 bg-clip-text text-transparent">
                  SecureX AI
                </h1>
                <p className="text-blue-200">Intelligent Academic Verification</p>
              </div>
            </div>

            <h2 className="text-4xl font-display font-bold mb-6 leading-tight">
              AI-Powered Certificate Authentication
            </h2>
            <p className="text-lg text-blue-100 mb-8 leading-relaxed">
              Experience the future of academic verification with SakshAI's intelligent 
              platform. Fast, secure, and trusted by educational institutions worldwide.
            </p>

            {/* Enhanced Trust Badges */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                <Verified className="h-5 w-5 text-green-400" />
                <span className="font-medium">AI-Powered Verification</span>
              </div>
              <div className="flex items-center space-x-3 p-3 rounded-lg bg-white/10 backdrop-blur-sm">
                <Shield className="h-5 w-5 text-blue-400" />
                <span className="font-medium">Military-Grade Security</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Enhanced Form Section */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-4 lg:p-8">
        <div className="w-full max-w-md space-y-6">
          {/* Enhanced Back to Home */}
          <Link
            to="/"
            className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800 transition-colors font-medium"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Homepage
          </Link>

          {/* Enhanced Form Card */}
          <Card className="border-blue-200/50 shadow-xl shadow-blue-100/50 hover:shadow-2xl hover:shadow-blue-200/30 transition-all duration-300">
            <CardHeader className="text-center pb-4">
              <div className="flex justify-center mb-4">
                <div className="w-16 h-16 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
                  <Brain className="h-8 w-8 text-white" />
                </div>
              </div>
              <CardTitle className="text-2xl font-display bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                {otpSent ? "Verify Your Identity" : "Welcome to SecureX AI"}
              </CardTitle>
              <p className="text-muted-foreground">
                {otpSent
                  ? "Enter the 6-digit code sent to your email"
                  : "Sign in to your account or create a new one"}
              </p>
            </CardHeader>

            <CardContent>
              {!otpSent ? (
                <Tabs
                  value={activeTab}
                  onValueChange={handleTabChange}
                  className="space-y-4"
                >
                  <TabsList className="grid w-full grid-cols-2 bg-blue-50/50">
                    <TabsTrigger 
                      value="login" 
                      className="data-[state=active]:bg-blue-500 data-[state=active]:text-white"
                    >
                      Login
                    </TabsTrigger>
                    <TabsTrigger 
                      value="register"
                      className="data-[state=active]:bg-purple-500 data-[state=active]:text-white"
                    >
                      Register
                    </TabsTrigger>
                  </TabsList>

                  {/* Enhanced Login Form */}
                  <TabsContent value="login" className="space-y-4">
                    <form onSubmit={handleSendOtp} className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="email" className="text-sm font-medium">
                          Email Address
                        </Label>
                        <div className="relative">
                          <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                          <Input
                            id="email"
                            type="email"
                            placeholder="Enter your email address"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="pl-10 border-blue-200 focus:border-blue-500"
                            required
                          />
                        </div>
                        {registeredEmail && (
                          <div className="flex items-center text-xs text-green-600">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Account created successfully! Please verify your email.
                          </div>
                        )}
                      </div>

                      <Button
                        type="submit"
                        className="w-full bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 transition-all duration-200 shadow-lg"
                        disabled={isLoading}
                      >
                        {isLoading ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                            Sending OTP...
                          </>
                        ) : (
                          "Send Verification Code"
                        )}
                      </Button>
                    </form>

                    <div className="relative my-6">
                      <div className="absolute inset-0 flex items-center">
                        <div className="w-full border-t border-gray-300"></div>
                      </div>
                      <div className="relative flex justify-center text-sm">
                        <span className="px-2 bg-white text-muted-foreground">
                          Secure & Encrypted
                        </span>
                      </div>
                    </div>
                  </TabsContent>

                  {/* Enhanced Register Form */}
                  <TabsContent value="register" className="space-y-4">
                    <form onSubmit={handleRegister} className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="firstName" className="text-sm font-medium">
                            First Name
                          </Label>
                          <Input
                            id="firstName"
                            placeholder="John"
                            value={firstName}
                            onChange={(e) => setFirstName(e.target.value)}
                            className="border-blue-200 focus:border-blue-500"
                            required
                            minLength={2}
                          />
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="lastName" className="text-sm font-medium">
                            Last Name
                          </Label>
                          <Input
                            id="lastName"
                            placeholder="Doe"
                            value={lastName}
                            onChange={(e) => setLastName(e.target.value)}
                            className="border-blue-200 focus:border-blue-500"
                            required
                            minLength={2}
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="registerEmail" className="text-sm font-medium">
                          Email Address
                        </Label>
                        <div className="relative">
                          <Mail className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                          <Input
                            id="registerEmail"
                            type="email"
                            placeholder="Enter your email address"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="pl-10 border-blue-200 focus:border-blue-500"
                            required
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="registerPassword" className="text-sm font-medium">
                          Password
                        </Label>
                        <div className="relative">
                          <Lock className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                          <Input
                            id="registerPassword"
                            type={showPassword ? "text" : "password"}
                            placeholder="Create a strong password"
                            value={password}
                            onChange={(e) => handlePasswordChange(e.target.value)}
                            className="pl-10 pr-10 border-blue-200 focus:border-blue-500"
                            required
                            minLength={8}
                          />
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="absolute right-2 top-1 h-8 w-8 p-0 hover:bg-blue-50"
                            onClick={() => setShowPassword(!showPassword)}
                          >
                            {showPassword ? (
                              <EyeOff className="h-4 w-4" />
                            ) : (
                              <Eye className="h-4 w-4" />
                            )}
                          </Button>
                        </div>

                        {/* Enhanced Password Strength Indicator */}
                        {password && (
                          <div className="space-y-2 pt-2">
                            <div className="flex items-center justify-between text-xs">
                              <span className="font-medium">Password Strength</span>
                              <span className={getPasswordStrengthColor(passwordStrength)}>
                                {passwordStrength >= 75
                                  ? "Strong"
                                  : passwordStrength >= 50
                                  ? "Medium"
                                  : "Weak"}
                              </span>
                            </div>
                            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className={`h-full transition-all duration-300 ${getPasswordStrengthBgColor(passwordStrength)}`}
                                style={{ width: `${passwordStrength}%` }}
                              />
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {password.length < 8 ? "At least 8 characters required" : 
                               "Include uppercase, numbers, and special characters for better security"}
                            </div>
                          </div>
                        )}
                      </div>

                      <Button
                        type="submit"
                        className="w-full bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 transition-all duration-200 shadow-lg"
                        disabled={isLoading || passwordStrength < 50}
                      >
                        {isLoading ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                            Creating Account...
                          </>
                        ) : (
                          "Create SecureX AI Account"
                        )}
                      </Button>
                    </form>
                  </TabsContent>
                </Tabs>
              ) : (
                // Enhanced OTP Verification Form
                <form onSubmit={handleVerifyOtp} className="space-y-4">
                  <div className="text-center mb-4">
                    <Verified className="h-12 w-12 text-green-500 mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground">
                      Verification code sent to
                    </p>
                    <p className="font-medium text-blue-600">{email}</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="otp" className="text-sm font-medium">
                      Enter 6-digit Code
                    </Label>
                    <Input
                      id="otp"
                      type="text"
                      placeholder="• • • • • •"
                      value={otp}
                      onChange={(e) => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                      onKeyPress={handleKeyPress}
                      className="text-center text-xl font-mono tracking-widest border-blue-200 focus:border-blue-500"
                      maxLength={6}
                      required
                    />
                    <p className="text-xs text-muted-foreground text-center">
                      Enter the 6-digit verification code sent to your email
                    </p>
                  </div>

                  <div className="flex space-x-2">
                    <Button
                      type="submit"
                      className="flex-1 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 shadow-lg"
                      disabled={isLoading || otp.length !== 6}
                    >
                      {isLoading ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                          Verifying...
                        </>
                      ) : (
                        "Verify & Continue"
                      )}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={resetOtpFlow}
                      disabled={isLoading}
                      className="border-blue-200 text-blue-600 hover:bg-blue-50"
                    >
                      Back
                    </Button>
                  </div>

                  <div className="text-center text-sm">
                    <button
                      type="button"
                      onClick={handleSendOtp}
                      className="text-blue-600 hover:text-blue-800 font-medium transition-colors"
                      disabled={isLoading}
                    >
                      Didn't receive code? Resend
                    </button>
                  </div>
                </form>
              )}

              {/* Enhanced Additional Links */}
              {!otpSent && (
                <div className="mt-6 text-center text-sm">
                  <Link to="/help" className="text-blue-600 hover:text-blue-800 font-medium transition-colors">
                    Need assistance? Contact our support team
                  </Link>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Enhanced Legal Links */}
          <div className="text-center text-xs text-muted-foreground space-y-1">
            <p>
              By continuing, you agree to our{" "}
              <Link to="/terms" className="text-blue-600 hover:text-blue-800 font-medium">
                Terms of Service
              </Link>{" "}
              and{" "}
              <Link to="/privacy" className="text-blue-600 hover:text-blue-800 font-medium">
                Privacy Policy
              </Link>
            </p>
            <p className="text-gray-500">
              © 2024 SakshAI. Intelligent Academic Verification Platform.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}