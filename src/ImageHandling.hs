{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module ImageHandling
    ( listFromImage, imageFromList, loadTwoImages
    ) where

import Codec.Picture
import qualified Codec.Picture.Extra as CPE
import System.FilePath (replaceExtension)
import Codec.Picture.Types
import qualified Data.Vector.Storable as V
import Data.Word
import Data.List

cleanImg :: Either String DynamicImage -> Image PixelRGB8
cleanImg (Left err) =  error "Could not read Image"
cleanImg (Right img)  =  convertRGB8 img

type Path = String

channels :: Int
channels = 3

getImgData :: Image PixelRGB8 -> [Float]
getImgData (Image _ _ imageData) = convertWF $ V.toList (imageData)

listFromImage :: Path -> IO ([Float], Int)
listFromImage path = do
  eimg <- readImage path
  let img = cleanImg eimg
      width = imageWidth img
      height = imageHeight img
      (transformedImg :: Image PixelRGB8) = if width /= height then cropAndResize img else img
      imgData = getImgData transformedImg
      size = imageWidth transformedImg
  return (imgData, size)

convertWF :: [Word8] -> [Float]
convertWF ws = [ (fromIntegral w) :: Float | w <- ws]

convertFW :: [Float] -> [Word8]
convertFW fs = [ (round f) :: Word8 | f <- fs]

imageFromList :: [Float] -> Int -> Path -> IO ()
imageFromList list size path = do
  let newList = convertFW list
  let image = ImageRGB8 (Image size size (V.fromList newList))
  saveJpgImage 100 path image


cropTest :: IO (String)
cropTest = do
  let filepath = "./obamabig.jpg"
  img <- readImage filepath

  let new_img = cropAndResize  (cleanImg img)
  saveJpgImage 100 "test.jpg"  ((ImageRGB8 new_img))
  return "test successful"

cropAndResize :: Image PixelRGB8 -> Image PixelRGB8
cropAndResize img = CPE.scaleBilinear (floor size) (floor size) (CPE.crop (floor x)  (floor y) (floor new_width) (floor new_height) img)
   where
     (size :: Float) = if new_width < new_height then new_width else new_height
     (new_width :: Float) = if width < height then width else height * (4/3)
     (new_height :: Float) = if height < width then height else width * (4/3)
     (width :: Float) = fromIntegral  $ imageWidth img
     (height :: Float) = fromIntegral $ imageHeight img
     (x :: Float) = if width < height then 0 else  ((width - (height * 4/3)) / 2)
     (y :: Float) = if width > height then 0 else  ((height - (width * 4/3)) / 2)

loadTwoImages :: FilePath -> FilePath -> IO (Int, [Float], [Float])
loadTwoImages path1 path2 = do
  eimg1 <- readImage path1
  eimg2 <- readImage path2
  let
    img1 = cleanImg eimg1
    img2 = cleanImg eimg2
    width1 = imageWidth img1
    width2 = imageWidth img2
    height1 = imageHeight img1
    height2 = imageHeight img2
    (transformedImg1 :: Image PixelRGB8) = if width1 /= height1 then cropAndResize img1 else img1
    (transformedImg2 :: Image PixelRGB8) = if width2 /= height2 then cropAndResize img2 else img2
    new_size1 = imageHeight transformedImg1
    new_size2 = imageHeight transformedImg2
    (transformedImg1New :: Image PixelRGB8) = if new_size1 > new_size2 then CPE.scaleBilinear new_size2 new_size2 transformedImg1 else transformedImg1
    (transformedImg2New :: Image PixelRGB8) = if new_size2 > new_size2 then CPE.scaleBilinear new_size1 new_size1 transformedImg2 else transformedImg2
    new_size = if new_size1 <= new_size2 then new_size1 else new_size2

  return (new_size, (getImgData transformedImg1New), (getImgData transformedImg2New))
