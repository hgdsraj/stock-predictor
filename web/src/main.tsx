import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "./components/ThemeProvider";
import { Layout } from "./components/Layout";
import { Home } from "./pages/Home";
import { Screener } from "./pages/Screener";
import { Ticker } from "./pages/Ticker";
import { Backtest } from "./pages/Backtest";
import { Jobs } from "./pages/Jobs";
import { Hypersearch } from "./pages/Hypersearch";
import { About } from "./pages/About";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 60_000,
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/screener" element={<Screener />} />
              <Route path="/ticker/:ticker" element={<Ticker />} />
              <Route path="/backtest" element={<Backtest />} />
              <Route path="/jobs" element={<Jobs />} />
              <Route path="/hypersearch" element={<Hypersearch />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={<Home />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </QueryClientProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
