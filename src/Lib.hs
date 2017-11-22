{-# LANGUAGE ScopedTypeVariables #-}
module Lib
     where

import qualified ImageHandling as Img
import qualified JsonHandling as Json
import Debug.Trace as Trace
import qualified Impl
import qualified Data.Vector as V
import System.Environment (getArgs)
import qualified TensorFlow.Core as TF

mainLib :: IO ()
mainLib = do
  --random seed
  let seed = 14783

  -- load images and commandline args
  [imgName, styleName, savePath, steps] <- getArgs
  let
    stylePath = "./style_images/" ++ styleName
    imgPath   = "./content_images/" ++ imgName
  (imgsize, imgList, styleList) <- Img.loadTwoImages imgPath stylePath
  let (stepsInt :: Int) =round $ read steps
  let imgVec = V.fromList  imgList :: V.Vector Float
  let styleVec = V.fromList styleList :: V.Vector Float

  -- read weights from json
  Trace.traceIO "Starting to load the weighs!"
  weights <- Json.readWeights "./weights/weights_flat.json"
  Trace.traceIO "Weights loaded! "
  Trace.traceIO ("size: " ++ (show imgsize) ++ "length imglist: " ++ (show $ length imgList) ++ "legnth styleList: " ++ (show $ length styleList) )

  --create and save output
  output <- TF.runSession (Impl.session weights imgsize imgVec styleVec stepsInt)
  let processed_output = V.toList output
  Img.imageFromList processed_output imgsize ("./output_images/" ++ savePath ++ ".jpg")
  putStrLn "executed"



