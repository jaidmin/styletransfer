{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Impl
     where
--standard imports
import Prelude 
import Control.Monad (replicateM_, forM_, when)
import Control.Monad.IO.Class (liftIO)
import qualified Data.Vector as V
import qualified Data.Int as I
import qualified Data.Text as T
import System.Environment (getArgs)
import qualified System.Random as R
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Char8 as B

--debug
import Debug.Trace as Trace

-- project internal imports
import qualified JsonHandling as Json
import qualified ImageHandling as Img
import qualified Vgg16 as Vgg16

-- tensorflow imports
import qualified TensorFlow.Tensor as TFten
import qualified TensorFlow.GenOps.Core as TFops (mul, gather, avgPool', div, conv2D,conv2D', maxPool, maxPool', add, square, squeeze, onesLike)
import qualified TensorFlow.Minimize as TFmin
import qualified TensorFlow.Ops as TFops
import qualified TensorFlow.Build as TFbuild
import qualified TensorFlow.Core as TF
import qualified TensorFlow.Variable  as TFvar
import qualified TensorFlow.Session as TFsess
import qualified TensorFlow.Gradient as TFgrad
import qualified TensorFlow.Types as TFtype
import Lens.Family2 (Lens', view, (&), (.~))

--type aliases
type TBF = TF.Tensor TF.Build Float
type TBI64= TF.Tensor TF.Build I.Int64
type TBI32= TF.Tensor TF.Build I.Int32

--helper function
shapeInBlock :: Int -> Int -> (Int,Int)
shapeInBlock shape 1  = (shape, 64)
shapeInBlock shape 2  = (shape `div`2 , 128)
shapeInBlock shape 3  = ((shape `div`2) `div` 2 , 256)
shapeInBlock shape 4  = (((shape `div`2) `div` 2) `div` 2 , 256 )
shapeInBlock shape 5  = ((((shape `div`2) `div` 2) `div` 2) `div` 2 , 512)

-- loss function 
mse :: TBF -> TBF -> TBF
mse pre targ = TFops.reduceMean $ TFops.square (pre `TFops.sub` targ)

contentloss :: TBF -> TBF -> TBF
contentloss imgTensorConv randTensorConv = mse imgTensorConv randTensorConv

batchFlatten :: TBF -> I.Int32 -> TBF
batchFlatten inp nr_channels = TFops.reshape transposed ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [nr_channels, -1]) :: TBI32)
  where
    transposed = (TFops.transpose inp ((TFops.constant (TF.Shape ([4] :: [I.Int64])) [3,0,1,2]) :: TBI32))
gramMatrix :: TBF -> Int -> Int -> TBF
gramMatrix inp shape block = (inp_flat `TFops.matMul` (TFops.transpose inp_flat ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [1,0]) :: TBI32))) `TFops.div`   (TFops.constant (TF.Shape ([1] :: [I.Int64])) [fromIntegral (nr_channels * shapeNow * shapeNow)]  )
  where
    (shapeNow, nr_channels) = shapeInBlock shape block
    inp_flat = batchFlatten inp (fromIntegral nr_channels)

styleloss :: Int -> Int -> TBF -> TBF -> TBF
styleloss shape block styleTensorConv randTensorConv = mse (gramMatrix styleTensorConv shape block) (gramMatrix randTensorConv shape block)

-- tensorflow session that is executed in mainLib
session :: Json.Weights -> Int -> V.Vector Float -> V.Vector Float -> Int -> TFsess.Session ( V.Vector Float)
session weights imgsize imgVector styleVector steps = do
  let
      seed = 1714654
      imgSize = imgsize
      (imgSizeI64 :: I.Int64) = fromIntegral imgSize
      randList = [122.5 | _ <- [1..(imgSize * imgSize * 3)]]

  (randVar :: TFvar.Variable Float) <- TFvar.initializedVariable (TFops.constant (TF.Shape ([1,imgSizeI64,imgSizeI64,3] :: [I.Int64]))  randList )

  let
      imgTensor = TFops.constant (TF.Shape ( [1,imgSizeI64,imgSizeI64,3] :: [I.Int64])) (V.toList imgVector)
      styleTensor = TFops.constant (TF.Shape ([1,imgSizeI64,imgSizeI64,3] :: [I.Int64])) (V.toList styleVector)
      randTensor   = (TFvar.readValue randVar)
      imgTensorConv3 = Vgg16.conv3_1 $ Vgg16.vgg16 weights imgTensor
      imgTensorConv4 = Vgg16.conv4_1 $ Vgg16.vgg16 weights imgTensor
      imgTensorConv5 = Vgg16.conv5_1 $ Vgg16.vgg16 weights imgTensor
      randTensorConv1 = Vgg16.conv1_1 $ Vgg16.vgg16 weights randTensor
      styleTensorConv1 = Vgg16.conv1_1 $ Vgg16.vgg16 weights styleTensor
      randTensorConv2 = Vgg16.conv2_1 $ Vgg16.vgg16 weights randTensor
      styleTensorConv2 = Vgg16.conv2_1 $ Vgg16.vgg16 weights styleTensor
      randTensorConv3 = Vgg16.conv3_1 $ Vgg16.vgg16 weights randTensor
      styleTensorConv3 = Vgg16.conv3_1 $ Vgg16.vgg16 weights styleTensor
      randTensorConv4 = Vgg16.conv4_1 $ Vgg16.vgg16 weights randTensor
      randTensorConv5 = Vgg16.conv5_1 $ Vgg16.vgg16 weights randTensor
      contentLoss = ((contentloss imgTensorConv4 randTensorConv4)) `TFops.div` 50
      styleLoss = ((styleloss imgSize 1 styleTensorConv1 randTensorConv1)) `TFops.add` (styleloss imgSize 2 styleTensorConv2 randTensorConv2)  `TFops.add` (styleloss imgSize 3 styleTensorConv3 randTensorConv3)
      thisloss = contentLoss `TFops.add`  styleLoss
      admcfg = TFmin.AdamConfig 0.1 0.9 0.999 1e-8

  updateStyle <- TFmin.minimizeWith (TFmin.adam' admcfg) styleLoss [randVar]
  let trainstepAll = TF.run updateStyle
  forM_ ([0..(steps `div` 3)] :: [Int]) $ \i -> do
    trainstepAll
    when  (i `mod` 100 == 0) $ do
      (styleLossRightNow :: V.Vector Float) <- TF.run styleLoss
      (contentLossRightNow :: V.Vector Float) <- TF.run contentLoss
      liftIO $ putStrLn $ "step: " ++ (show i) ++ " styleloss: " ++ (show styleLossRightNow) ++ " contentloss: " ++ (show contentLossRightNow)


  updateAll <- TFmin.minimizeWith (TFmin.adam' admcfg) thisloss [randVar]
  let trainstepAll = TF.run updateAll
  forM_ ([0..(steps)] :: [Int]) $ \i -> do
    trainstepAll
    when  (i `mod` 100 == 0) $ do
      (styleLossRightNow :: V.Vector Float) <- TF.run styleLoss
      (contentLossRightNow :: V.Vector Float) <- TF.run contentLoss
      liftIO $ putStrLn $ "step: " ++ (show i) ++ " styleloss: " ++ (show styleLossRightNow) ++ " contentloss: " ++ (show contentLossRightNow)

  let output = (TFvar.readValue randVar)
  TF.run output


