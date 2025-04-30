import React from "react";
import ShootingStar from "./ShootingStar";

const StarBanner = ({ numberOfStars }) => {
  const stars = Array.from({ length: numberOfStars }).map((_, index) => {
    const top = Math.random() * 100;
    const left = Math.random() * 100;

    return (
      <div
        key={index}
        className="absolute size-px rounded-full bg-white opacity-80"
        style={{
          top: `${top}%`,
          left: `${left}%`,
        }}
      />
    );
  });

  return (
    <div className="fixed inset-0 w-full h-full pointer-events-none z-0">
      {stars}
      <ShootingStar />
    </div>
  );
};

export default StarBanner; 