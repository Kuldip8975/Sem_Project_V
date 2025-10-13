import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Shield,
  FileSearch,
  CheckCircle,
  Users,
  Building2,
  ArrowRight,
  Zap,
  Globe,
  Lock,
  BadgeCheck,
  Clock,
  Upload,
  GraduationCap,
  BookOpen,
  Award,
  Star,
  Heart,
  TrendingUp,
  ShieldCheck,
  FileText,
  Database,
  Cpu,
  Brain,
  Rocket,
  Target,
  Medal,
  Crown,
  Lightbulb,
  Sparkles,
  Bot,
  Eye,
  Scan,
  Mail,
  Phone,
  MapPin,
  ExternalLink,
  Github,
  Twitter,
  Linkedin,
} from "lucide-react";

const quickLinks = [
  { href: "/", label: "Home" },
  { href: "/verify", label: "Verify Document" },
  { href: "/help", label: "Help Center" },
  { href: "/login", label: "Login" },
];

const resources = [
  { href: "/help", label: "Documentation" },
  { href: "/help", label: "API Reference" },
  { href: "/help", label: "Integration Guide" },
  { href: "/help", label: "Status Page" },
];

const legal = [
  { href: "/privacy", label: "Privacy Policy" },
  { href: "/terms", label: "Terms of Service" },
  { href: "/help", label: "Security Policy" },
  { href: "/help", label: "Cookie Policy" },
];

function Footer() {
  return (
    <footer className="bg-secondary text-white">
      <div className="container py-16">
        {/* Main Footer */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-hero-gradient">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h3 className="font-display text-xl font-bold">SecureX AI</h3>
                <p className="text-sm text-white/70">Intelligent Document Verification</p>
              </div>
            </div>
            <p className="text-white/80">
              Securing academic credentials with cutting-edge verification
              technology.
            </p>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary" className="bg-white/20 text-white">
                Trusted Platform
              </Badge>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-3">
              {quickLinks.map((link) => (
                <li key={link.href + link.label}>
                  <Link
                    to={link.href}
                    className="text-white/80 hover:text-white transition-colors"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-semibold mb-4">Resources</h4>
            <ul className="space-y-3">
              {resources.map((link) => (
                <li key={link.href + link.label}>
                  <Link
                    to={link.href}
                    className="text-white/80 hover:text-white transition-colors inline-flex items-center"
                  >
                    {link.label}
                    <ExternalLink className="h-3 w-3 ml-1" />
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="font-semibold mb-4">Contact</h4>
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <MapPin className="h-4 w-4 text-white/60" />
                <span className="text-sm text-white/80">
                  SecureX AI Team
                  <br />
                  Kolkata, West-Bengal
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <Phone className="h-4 w-4 text-white/60" />
                <span className="text-sm text-white/80">+91 8335867482</span>
              </div>
              <div className="flex items-center space-x-2">
                <Mail className="h-4 w-4 text-white/60" />
                <span className="text-sm text-white/80">
                  support@securexai.com
                </span>
              </div>
            </div>

            {/* Social Links */}
            <div className="flex items-center space-x-2 mt-4">
              <Button
                variant="ghost"
                size="icon"
                className="text-white/60 hover:text-white"
              >
                <Twitter className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="text-white/60 hover:text-white"
              >
                <Linkedin className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="text-white/60 hover:text-white"
              >
                <Github className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-white/20 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="text-sm text-white/60">
              © 2025 SecureX AI. All rights reserved.
            </div>
            <div className="flex flex-wrap gap-6">
              {legal.map((link) => (
                <Link
                  key={link.href + link.label}
                  to={link.href}
                  className="text-sm text-white/60 hover:text-white transition-colors"
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-cyan-100">
      {/* Navigation */}
      <nav className="border-b bg-white/95 backdrop-blur shadow-lg sticky top-0 z-50">
        <div className="container flex h-20 items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 shadow-lg">
              <Bot className="h-7 w-7 text-white" />
            </div>
            <div>
              <h1 className="font-display text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                SecureX AI
              </h1>
              <p className="text-sm text-gray-600 font-medium">Intelligent Document Verification</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <Link to="/login">
              <Button variant="ghost" className="text-gray-700 hover:text-purple-600 font-semibold">Login</Button>
            </Link>
            <Link to="/verify">
              <Button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-lg hover:shadow-xl transition-all">
                Verify Now
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden py-24">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 to-blue-600/10 transform skew-y-3 scale-105"></div>
        <div className="container relative z-10">
          <div className="text-center max-w-5xl mx-auto space-y-8">
            <div className="space-y-6">
              <div className="inline-flex items-center space-x-2 bg-purple-100 text-purple-700 px-4 py-2 rounded-full text-sm font-medium mb-4">
                <Sparkles className="h-4 w-4" />
                <span>AI-Powered Document Authentication</span>
              </div>
              
              <h1 className="text-6xl font-display font-bold text-gray-900 leading-tight">
                Verify with
                <span className="block bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                  SecureX AI Intelligence
                </span>
              </h1>
              
              <p className="text-2xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
                Advanced AI technology to authenticate academic documents, certificates, 
                and transcripts with precision and speed
              </p>
            </div>

            <div className="flex items-center justify-center space-x-6 pt-8">
              <Link to="/verify">
                <Button size="xl" className="text-lg px-10 py-7 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-2xl hover:scale-105 transition-transform">
                  <Scan className="mr-3 h-6 w-6" />
                  Start Verification
                  <ArrowRight className="ml-3 h-5 w-5" />
                </Button>
              </Link>
              <Link to="/demo">
                <Button variant="outline" size="xl" className="text-lg px-10 py-7 border-2 border-gray-300 text-gray-700 hover:border-purple-500 hover:text-purple-600">
                  <Eye className="mr-3 h-6 w-6" />
                  View Demo
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 bg-white/50">
        <div className="container">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Why SecureX AI Stands Out</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Cutting-edge features designed for modern document verification needs
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8 mb-16">
            <Card className="border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 group hover:scale-105">
              <CardContent className="p-8 text-center relative overflow-hidden">
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-purple-500 to-blue-600 rounded-bl-full opacity-10"></div>
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-r from-purple-500 to-blue-600 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                  <Brain className="h-10 w-10 text-white" />
                </div>
                <h3 className="font-bold text-2xl mb-4 text-gray-900">AI-Powered Analysis</h3>
                <ul className="text-left space-y-3 text-gray-600">
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-purple-500 flex-shrink-0" />
                    <span>Smart pattern recognition</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-purple-500 flex-shrink-0" />
                    <span>Machine learning algorithms</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-purple-500 flex-shrink-0" />
                    <span>Continuous improvement</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 group hover:scale-105">
              <CardContent className="p-8 text-center relative overflow-hidden">
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-bl-full opacity-10"></div>
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                  <Zap className="h-10 w-10 text-white" />
                </div>
                <h3 className="font-bold text-2xl mb-4 text-gray-900">Lightning Fast</h3>
                <ul className="text-left space-y-3 text-gray-600">
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-cyan-500 flex-shrink-0" />
                    <span>Instant verification results</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-cyan-500 flex-shrink-0" />
                    <span>Real-time processing</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-cyan-500 flex-shrink-0" />
                    <span>Batch processing support</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card className="border-0 shadow-2xl hover:shadow-3xl transition-all duration-500 group hover:scale-105">
              <CardContent className="p-8 text-center relative overflow-hidden">
                <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-600 rounded-bl-full opacity-10"></div>
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-r from-green-500 to-emerald-600 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                  <ShieldCheck className="h-10 w-10 text-white" />
                </div>
                <h3 className="font-bold text-2xl mb-4 text-gray-900">Military Grade Security</h3>
                <ul className="text-left space-y-3 text-gray-600">
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
                    <span>End-to-end encryption</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
                    <span>Secure data handling</span>
                  </li>
                  <li className="flex items-center space-x-3">
                    <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0" />
                    <span>Privacy first approach</span>
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20">
        <div className="container">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">How SecureX AI Works</h2>
            <p className="text-xl text-gray-600">Three simple steps to verify any document</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-20">
            {[
              { icon: Upload, number: "01", title: "Upload Document", desc: "Drag & drop your document", color: "from-purple-500 to-blue-600" },
              { icon: Cpu, number: "02", title: "AI Processing", desc: "SecureX AI analyzes the content", color: "from-cyan-500 to-blue-600" },
              { icon: BadgeCheck, number: "03", title: "Get Verified", desc: "Receive authenticity report", color: "from-green-500 to-emerald-600" }
            ].map((step, index) => (
              <div key={index} className="text-center group">
                <div className="relative mb-8">
                  <div className={`w-24 h-24 rounded-3xl bg-gradient-to-r ${step.color} flex items-center justify-center mx-auto shadow-2xl group-hover:scale-110 transition-transform`}>
                    <step.icon className="h-12 w-12 text-white" />
                  </div>
                  <div className="absolute -top-2 -right-2 w-12 h-12 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg">
                    {step.number}
                  </div>
                </div>
                <h3 className="font-bold text-2xl mb-3 text-gray-900">{step.title}</h3>
                <p className="text-gray-600 text-lg">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-20 bg-gradient-to-br from-gray-900 to-purple-900 text-white">
        <div className="container text-center">
          <h2 className="text-5xl font-bold mb-6">Experience SecureX AI Today</h2>
          <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto">
            Join the future of document verification with our intelligent AI-powered platform
          </p>
          
          <div className="flex items-center justify-center space-x-6 mb-8">
            <Link to="/verify">
              <Button size="xl" className="text-lg px-12 py-7 bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white shadow-2xl">
                <Scan className="mr-3 h-6 w-6" />
                Start Verification
              </Button>
            </Link>
            <Link to="/demo">
              <Button variant="outline" size="xl" className="text-lg px-12 py-7 border-2 border-white text-white hover:bg-white/10">
                <Eye className="mr-3 h-6 w-6" />
                View Demo
              </Button>
            </Link>
          </div>
          
          <div className="flex items-center justify-center space-x-8 text-purple-300">
            <div className="flex items-center space-x-2">
              <ShieldCheck className="h-5 w-5" />
              <span>Secure & Private</span>
            </div>
            <div className="flex items-center space-x-2">
              <Zap className="h-5 w-5" />
              <span>Fast Processing</span>
            </div>
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>AI Powered</span>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <Footer />
    </div>
  );
}