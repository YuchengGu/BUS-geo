#pragma once
#include "RandomWalk.h"
#include <ITK-5.4/itkImage.h>
#include <ITK-5.4/itkImageFileReader.h>
#include <ITK-5.4/itkExtractImageFilter.h>
#include <ITK-5.4/itkMetaImageIOFactory.h>
#include <itkCastImageFilter.h>
#include <itkBoxMeanImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkSqrtImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkSubtractImageFilter.h>   
#include <itkAddImageFilter.h>   
#include <itkDivideImageFilter.h> 
#include <itkImageFileWriter.h> 
#include <itkClampImageFilter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>

class Image
{
public:
    using Image2DType = itk::Image<float, 2>;
    Image2DType::Pointer GetImage() const { return m_image; }

    explicit Image(const std::string& mhaPath, int sliceZ = 0)
    {
        //注册 IO 
        itk::MetaImageIOFactory::RegisterOneFactory();

        // 读
        using Image3DType = itk::Image<float, 3>;
        using Reader3DType = itk::ImageFileReader<Image3DType>;
        Reader3DType::Pointer reader3D = Reader3DType::New();
        reader3D->SetFileName(mhaPath);
        reader3D->Update();
        Image3DType::Pointer vol = reader3D->GetOutput();


        Image3DType::RegionType inRegion = vol->GetLargestPossibleRegion();
        Image3DType::SizeType   inSize = inRegion.GetSize();
        Image3DType::IndexType  inIndex = inRegion.GetIndex();

        // 只改 z 方向
        inIndex[2] = sliceZ;
        inSize[2] = 0;      

        using ExtractType = itk::ExtractImageFilter<Image3DType, Image2DType>;
        ExtractType::Pointer extractor = ExtractType::New();
        extractor->SetDirectionCollapseToSubmatrix();  // 把第 3 维去掉
        extractor->SetInput(vol);
        extractor->SetExtractionRegion({ inIndex, inSize });
        extractor->Update();


        m_image = extractor->GetOutput();
        m_image->DisconnectPipeline();
    }





    //Shannon 熵  (0-255 直方图，归一化到 0-1) 
    inline
    float ShannonEntropy(int bins = 256)
    {
        const auto* buf = m_image->GetBufferPointer();
        const size_t  N = m_image->GetLargestPossibleRegion().GetNumberOfPixels();

        std::vector<int> hist(bins, 0);
        for (size_t i = 0; i < N; ++i)
        {
            int idx = static_cast<int>(buf[i]) * bins / 256; // [0,bins-1]
            idx = std::clamp(idx, 0, bins - 1);
            ++hist[idx];
        }
        float H = 0.0f;
        for (int c : hist)
        {
            if (c == 0) continue;
            float p = static_cast<float>(c) / N;
            H -= p * std::log2(p);
        }
        return H; // bit
    }

    ///对比度：整图灰度标准差
    inline
    float Contrast()
    {
        const auto* buf = m_image->GetBufferPointer();
        const size_t  N = m_image->GetLargestPossibleRegion().GetNumberOfPixels();

        double mean = std::accumulate(buf, buf + N, 0.0) / N;
        double var = 0.0;
        for (size_t i = 0; i < N; ++i)
        {
            double d = buf[i] - mean;
            var += d * d;
        }
        return static_cast<float>(std::sqrt(var / N));
    }

    //SpeckleIdx：亮区（mean>10）局部 CV 平均
    //win=9 ，dark_abs=10
    inline
        float SpeckleIndex(int win = 9,
            float darkAbs = 10.0f)
    {
        const int h = m_image->GetLargestPossibleRegion().GetSize()[1];
        const int w = m_image->GetLargestPossibleRegion().GetSize()[0];
        const auto* src = m_image->GetBufferPointer();

        int rad = win / 2;
        long tot = 0;
        double cv = 0.0;

        for (int y = rad; y < h - rad; ++y)
            for (int x = rad; x < w - rad; ++x)
            {
                double s0 = 0.0, s1 = 0.0;
                int cnt = 0;
                for (int dy = -rad; dy <= rad; ++dy)
                    for (int dx = -rad; dx <= rad; ++dx)
                    {
                        float v = src[(y + dy) * w + (x + dx)];
                        s0 += v;
                        s1 += v * v;
                        ++cnt;
                    }
                double mean = s0 / cnt;
                if (mean <= darkAbs) continue;
                double std = std::sqrt(s1 / cnt - mean * mean);
                cv += std / (mean + 1e-8f);
                ++tot;
            }
        return tot ? static_cast<float>(cv / tot) : 0.0f;
    }





    // 置信图，乘以映射后的亮度权重 
    inline FloatImageType::Pointer
        ConfidenceMap()
    {
        const auto* buf = m_image->GetBufferPointer();
        int W = m_image->GetLargestPossibleRegion().GetSize()[0];
        int H = m_image->GetLargestPossibleRegion().GetSize()[1];


        float Imin = *std::min_element(buf, buf + W * H);
        float Imax = *std::max_element(buf, buf + W * H);
        float scale = 1.0f / (Imax - Imin + 1e-8f);
        std::vector<float> I(W * H);
        for (size_t i = 0; i < W * H; ++i) I[i] = (buf[i] - Imin) * scale;


        auto fImg = FloatImageType::New();
        fImg->SetRegions(m_image->GetLargestPossibleRegion());
        fImg->Allocate();
        std::copy(I.begin(), I.end(), fImg->GetBufferPointer());


        return RandomWalk(fImg, 25.0, 0.001);
    }


    inline float
        WeightedConfidenceMap()
    {
        float lowThr = 10;
        auto confImg = ConfidenceMap();  

        const size_t N = m_image->GetLargestPossibleRegion().GetNumberOfPixels();
        std::vector<float> tmp(N);
        std::copy(m_image->GetBufferPointer(),
            m_image->GetBufferPointer() + N,
            tmp.begin());
        std::nth_element(tmp.begin(),
            tmp.begin() + static_cast<size_t>(0.90 * N),
            tmp.end());
        float p90 = tmp[static_cast<size_t>(0.90 * N)];
        if (p90 <= lowThr) p90 = lowThr + 1.0f;

        const float* src = m_image->GetBufferPointer();
        const float* confP = confImg->GetBufferPointer();
        double numer = 0.0, denom = 0.0;
        for (size_t i = 0; i < N; ++i)
        {
            float v = src[i];
            float t = (v - lowThr) / (p90 - lowThr);
            t = std::clamp(t, 0.0f, 1.0f);
            float bright = std::sqrt(t);
            float x = confP[i] * bright;
			float y = confP[i];
            numer += x;
            denom += y;
        }
        return static_cast<float>(numer / (denom + 1e-8f));
    }


    float evaluate()
    {
        float D = WeightedConfidenceMap();
        float E = ShannonEntropy();
        float C = Contrast();
        float S = SpeckleIndex();

		float d, e, c, s;
		d = (D - 0.1059f) / (0.9156f - 0.1059f);    
		e = (E - 1.6738f) / (7.794f - 1.6738f);         
		c = 1 - (abs(C - 48.705f)) / (71.07f - 18.34f);   
		s = (0.529f - S) / (0.529f - 0.085f);   
        
        d = 0.48330939f * d;
		e = 0.16198774f * e;
		c = 0.07935044f * c;
		s = 0.27535243f * s;


		const float x[4] = { d, e, c, s };

        float d_pos = 0.0, d_neg = 0.0;
        for (int i = 0; i < 4; ++i) {
            d_pos += (x[i] - z_pos[i]) * (x[i] - z_pos[i]);
            d_neg += (x[i] - z_neg[i]) * (x[i] - z_neg[i]);
        }
        d_pos = std::sqrt(d_pos);
        d_neg = std::sqrt(d_neg);

        float score = d_neg / (d_pos + d_neg + 1e-8);  

        return score;
	}



private:

    Image2DType::Pointer m_image;

    const float z_pos[4] = { 0.48330939f, 0.16198774f, 0.07934357f, 0.27535243f };
    const float z_neg[4] = { 0.0f, 0.0f, 0.03365519f, 0.0f };

};