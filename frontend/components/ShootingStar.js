"use client";
import React, { useState, useEffect } from "react";

const ShootingStar = () => {
  const [position, setPosition] = useState({ top: "40%", left: "10%" });
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      const top = Math.random() * 100;
      const left = Math.random() * 100;
      setPosition({ top: `${top}%`, left: `${left}%` });
      setIsVisible(true);

      setTimeout(() => {
        setIsVisible(false);
      }, 1000);
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      {isVisible && (
        <svg
          width="67"
          height="68"
          viewBox="0 0 67 68"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          style={{ top: position.top, left: position.left, position: 'absolute' }}
          className="z-10 animate-shooting"
        >
          <g filter="url(#filter0_f_928_3155)">
            <circle cx="59" cy="8" r="2" fill="white" fillOpacity="1" />
          </g>
          <path
            d="M59.3535 8.35355L0.353512 67.3535"
            stroke="url(#paint0_linear_928_3155)"
            strokeOpacity="1"
            strokeWidth="0.5"
            strokeLinecap="round"
          />
          <defs>
            <filter
              id="filter0_f_928_3155"
              x="51"
              y="0"
              width="16"
              height="16"
              filterUnits="userSpaceOnUse"
              colorInterpolationFilters="sRGB"
            >
              <feFlood floodOpacity="0" result="BackgroundImageFix" />
              <feBlend
                mode="normal"
                in="SourceGraphic"
                in2="BackgroundImageFix"
                result="shape"
              />
              <feGaussianBlur
                stdDeviation="3"
                result="effect1_foregroundBlur_928_3155"
              />
            </filter>
            <linearGradient
              id="paint0_linear_928_3155"
              x1="58.6464"
              y1="7.64645"
              x2="-0.353557"
              y2="66.6464"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="white" />
              <stop offset="1" stopColor="#1D1D20" />
            </linearGradient>
          </defs>
        </svg>
      )}
      <style jsx global>{`
        @keyframes shooting {
          0% {
            opacity: 0.5;
          }
          50% {
            opacity: 1;
          }
          100% {
            transform: translateX(500px) translateY(-500px);
            opacity: 0;
          }
        }
        .animate-shooting {
          animation: shooting 1s forwards;
        }
      `}</style>
    </div>
  );
};

export default ShootingStar; 