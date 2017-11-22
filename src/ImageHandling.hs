{-# LANGUAGE FlexibleContexts #-}

module ImageHandling
    ( listFromImage, imageFromList
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

listFromImage :: Path -> IO ([[[Float]]], Int)
listFromImage path = do
  eimg <- readImage path
  let img = cleanImg eimg
      oneDimList = case img of (Image _ _ imageData) ->  V.toList imageData
      width = case img of (Image width _ _) ->  width
      height = case img of (Image _ height _) ->  height
      (transformedList, size) = transformImg width height (convertWF oneDimList)
      threeDimList = ((splitEvery height) . (splitEvery channels)) transformedList
  return (threeDimList, size)

splitEvery :: Int -> [a] -> [[a]]
splitEvery _ [] = []
splitEvery x ls = take x ls : splitEvery x (drop x ls)

convertWF :: [Word8] -> [Float]
convertWF ws = [ (fromIntegral w) :: Float | w <- ws]

convertFW :: [Float] -> [Word8]
convertFW fs = [ (round f) :: Word8 | f <- fs]

imageFromList :: [Float] -> Int -> Path -> IO ()
imageFromList list size path = do
  let newList = convertFW list
<<<<<<< HEAD
  let image = ImageRGB8 (Image 448 448 (V.fromList newList))
=======
  let image = ImageRGB8 (Image size size (V.fromList newList))
>>>>>>> be98fea54aa24c1a1fba0b98afbc1cf9e397f426
  savePngImage path image

transformImg :: Int -> Int -> [Float] -> ([Float], Int)
transformImg width height img
  | (3/4) < ratio && ratio <   1   = ((resizeWRT width dim img), width)
  |         ratio ==  1            = (img, width)
  |   1   < ratio && ratio < (4/3) = ((resizeWRT height dim img), height)
  | otherwise                      = cropAndResize dim img
  where
    ratio = ((fromIntegral width) / (fromIntegral height)) :: Float
    dim = (width, height) :: (Int, Int)


cropAndResize :: (Int, Int) -> [Float] -> ([Float], Int)
cropAndResize (width, height) list
  | width < height = (resizeWRT width (width, height) (convertWF cropHeight), width)
  | width > height = (resizeWRT height (width, height) (convertWF cropWidth), height)
  where
    w = fromIntegral width
    h = fromIntegral height
    x = round ((w - (h * 4/3)) / 2)
    y = round ((h - (w * 4/3)) / 2)
    img = ImageRGB8 (Image width height (V.fromList $ convertFW list))
    getList (Image _ _ x) = V.toList x
    cropHeight = getList $ CPE.crop 0 y width (round (w * 4/3)) (convertRGB8 img)
    cropWidth  = getList $ CPE.crop x 0 (round (h * 4/3)) height (convertRGB8 img)

resizeWRT :: Int -> (Int, Int) -> [Float] -> [Float]
resizeWRT size (width, height) list = convertWF $ getList (CPE.scaleBilinear size size img)
  where
    img = Image width height (V.fromList $ convertFW list)
    getList (Image _ _ x) = V.toList x



















