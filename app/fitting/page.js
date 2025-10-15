'use client';
import { useState } from 'react';
import Image from 'next/image';

export default function FittingPage() {
  const [personImg, setPersonImg] = useState(null);
  const [clothImg, setClothImg] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e, type) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = () => {
      if (type === 'person') setPersonImg(reader.result);
      else setClothImg(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleTryOn = async () => {
    if (!personImg || !clothImg) {
      alert('이미지를 모두 선택해주세요!');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('/api/tryon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person: personImg, cloth: clothImg })
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.result);
      } else {
        alert('에러: ' + data.error);
      }
    } catch (error) {
      alert('에러 발생: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">🎨 가상 피팅</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">1. 사람 이미지</h2>
            <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, 'person')} className="mb-4" />
            {personImg && <img src={personImg} alt="Person" className="w-full rounded" />}
          </div>

          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-semibold mb-4">2. 의상 이미지</h2>
            <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, 'cloth')} className="mb-4" />
            {clothImg && <img src={clothImg} alt="Cloth" className="w-full rounded" />}
          </div>
        </div>

        <div className="text-center mb-8">
          <button 
            onClick={handleTryOn} 
            disabled={loading}
            className="bg-blue-600 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? '처리 중... (30초~1분)' : '가상 착용 실행!'}
          </button>
        </div>

        {result && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-2xl font-semibold mb-4 text-center">✅ 결과</h2>
            <img src={result} alt="Result" className="w-full max-w-2xl mx-auto rounded" />
          </div>
        )}
      </div>
    </div>
  );
}