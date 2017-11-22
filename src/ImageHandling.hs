module ImageHandling
    ( listFromImage, imageFromList
    ) where

import Codec.Picture
import System.FilePath (replaceExtension)
import qualified Codec.Picture.Types as M
import qualified Data.Vector.Storable as V
import Data.Word
import Data.List

cleanImg :: Either String DynamicImage -> Image PixelRGB8
cleanImg (Left err) =  error "Could not read Image"
cleanImg (Right img)  =  convertRGB8 img

type Path = String

channels :: Int
channels = 3

listFromImage :: Path -> IO [[[Float]]]
listFromImage path = do
  eimg <- readImage path
  let img = cleanImg eimg
  let list = case img of (Image _ _ imageData) ->  V.toList imageData
  let width = case img of (Image width _ _) ->  width
  let height = case img of (Image _ height _) ->  height
  let newList = ((splitEvery height) . (splitEvery channels) . convertWF) list
  return newList

splitEvery :: Int -> [a] -> [[a]]
splitEvery _ [] = []
splitEvery x ls = take x ls : splitEvery x (drop x ls)

convertWF :: [Word8] -> [Float]
convertWF ws = [ (fromIntegral w) :: Float | w <- ws]

convertFW :: [Float] -> [Word8]
convertFW fs = [ (round f) :: Word8 | f <- fs]

imageFromList :: [Float] -> Path -> IO ()
imageFromList list path = do 
 -- let width = length list
 -- let height = length $ head list
  let newList = convertFW list
  let image = ImageRGB8 (Image 448 448 (V.fromList newList))
  savePngImage path image
