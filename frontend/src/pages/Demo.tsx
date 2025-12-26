import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

export default function Demo() {
  return (
    <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center px-4">

      {/* Back Button */}
      <Link
        to="/"
        className="flex items-center text-white mb-6 hover:text-indigo-400"
      >
        <ArrowLeft className="mr-2" />
        Back to Home
      </Link>

      {/* Heading */}
      <h1 className="text-4xl font-bold text-white mb-8 text-center">
        SecureX AI – Live Demo
      </h1>

      {/* YouTube Video - MEDIUM SIZE */}
      <div className="w-full max-w-5xl aspect-video rounded-xl overflow-hidden shadow-2xl border border-white/20">
        <iframe
          className="w-full h-full"
          src="https://www.youtube-nocookie.com/embed/JquiqI_Sw0o?vq=hd1080"
          title="Live Demo"
          frameBorder="0"
          allow="accelerometer; encrypted-media; picture-in-picture"
          allowFullScreen
        />
      </div>

    </div>
  );
}
