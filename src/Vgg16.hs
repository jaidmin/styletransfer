{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Vgg16
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


data Vgg16 = Vgg16 {
              conv1_1 :: TBF ,
              conv1_2 :: TBF ,
              pool1   :: TBF ,
              conv2_1 :: TBF ,
              conv2_2 :: TBF ,
              pool2   :: TBF ,
              conv3_1 :: TBF ,
              conv3_2 :: TBF ,
              conv3_3 :: TBF ,
              pool3   :: TBF ,
              conv4_1 :: TBF ,
              conv4_2 :: TBF ,
              conv4_3 :: TBF ,
              pool4   :: TBF ,
              conv5_1 :: TBF ,
              conv5_2 :: TBF ,
              conv5_3 :: TBF ,
              pool5   :: TBF
                   }

vgg16 :: Json.Weights -> TBF -> Vgg16
vgg16 weightDict input = Vgg16 {
                        conv1_1 = conv1_1_out,
                        conv1_2 = conv1_2_out,
                        pool1   = pool1,
                        conv2_1 = conv2_1_out,
                        conv2_2 = conv2_2_out,
                        pool2   = pool2,
                        conv3_1 = conv3_1_out,
                        conv3_2 = conv3_2_out,
                        conv3_3 = conv3_3_out,
                        pool3   = pool3,
                        conv4_1 = conv4_1_out,
                        conv4_2 = conv4_2_out,
                        conv4_3 = conv4_3_out,
                        pool4   = pool4,
                        conv5_1 = conv5_1_out,
                        conv5_2 = conv5_2_out,
                        conv5_3 = conv5_3_out,
                        pool5   = pool5
                               }
  -- preprocessing
  where

 -------------------------
 -- normalizing the pixels to imagenet mean (because vgg was trained that way)
 -------------------------

    (mean :: TBF)              = TFops.constant (TF.Shape ([1,1,1,3] :: [I.Int64])) [123.68, 116.779, 103.939]
    (normalized  :: TBF)       = input `TFops.sub` mean

 -------------------------------------
-- block 1
  ------------------------------------

  -- conv1_1
    (conv1_1_W :: TBF)         = TFops.constant (TF.Shape ([3,3,3,64])) ( Json.conv1_1_W weightDict)
    (conv1_1_conv :: TBF)      = TFops.conv2D'
                                 (  (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                  . (TFbuild.opAttr "padding" .~ (B.pack "SAME" )  )
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 normalized conv1_1_W
    (conv1_1_b :: TBF)         = TFops.constant (TF.Shape ([64])) (Json.conv1_1_b weightDict)
    (conv1_1_bias :: TBF)      = TFops.add conv1_1_conv conv1_1_b
    (conv1_1_out :: TBF)       = TFops.relu conv1_1_bias

  -- conv1_2
    (conv1_2_W :: TBF)         = TFops.constant (TF.Shape ([3,3,64,64])) ( Json.conv1_2_W weightDict)
    (conv1_2_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 conv1_1_out conv1_2_W
    (conv1_2_b :: TBF)         = TFops.constant (TF.Shape ([64])) (Json.conv1_2_b weightDict)
    (conv1_2_bias :: TBF)      = TFops.add conv1_2_conv conv1_2_b
    (conv1_2_out :: TBF)       = TFops.relu conv1_2_bias

  -- pool1
    (pool1 :: TBF)             = TFops.avgPool'
                                 (
                                   (TFbuild.opAttr "ksize"   .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "strides" .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME"))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 )
                                 conv1_2_out

 -------------------------------------
-- block 2
  ------------------------------------

  -- conv2_1
    (conv2_1_W :: TBF)         = TFops.constant (TF.Shape ([3,3,64,128])) ( Json.conv2_1_W weightDict)
    (conv2_1_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 pool1 conv2_1_W
    (conv2_1_b :: TBF)         = TFops.constant (TF.Shape ([128])) (Json.conv2_1_b weightDict)
    (conv2_1_bias :: TBF)      = TFops.add conv2_1_conv conv2_1_b
    (conv2_1_out :: TBF)       = TFops.relu conv2_1_bias

  -- conv2_2
    (conv2_2_W :: TBF)         = TFops.constant (TF.Shape ([3,3,128,128])) ( Json.conv2_2_W weightDict)
    (conv2_2_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 conv2_1_out conv2_2_W
    (conv2_2_b :: TBF)         = TFops.constant (TF.Shape ([128])) (Json.conv2_2_b weightDict)
    (conv2_2_bias :: TBF)      = TFops.add conv2_2_conv conv2_2_b
    (conv2_2_out :: TBF)       = TFops.relu conv2_2_bias

  -- pool2
    (pool2 :: TBF)             = TFops.avgPool'
                                 (
                                   (TFbuild.opAttr "ksize"   .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "strides" .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME"))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 )
                                 conv2_2_out

 -------------------------------------
-- block 3
  -------------------------------------

  -- conv3_1
    (conv3_1_W :: TBF)         = TFops.constant (TF.Shape ([3,3,128,256])) ( Json.conv3_1_W weightDict)
    (conv3_1_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 pool2 conv3_1_W
    (conv3_1_b :: TBF)         = TFops.constant (TF.Shape ([256])) (Json.conv3_1_b weightDict)
    (conv3_1_bias :: TBF)      = TFops.add conv3_1_conv conv3_1_b
    (conv3_1_out :: TBF)       = TFops.relu conv3_1_bias

  -- conv3_2
    (conv3_2_W :: TBF)         = TFops.constant (TF.Shape ([3,3,256,256])) ( Json.conv3_2_W weightDict)
    (conv3_2_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 conv3_1_out conv3_2_W
    (conv3_2_b :: TBF)         = TFops.constant (TF.Shape ([256])) (Json.conv3_2_b weightDict)
    (conv3_2_bias :: TBF)      = TFops.add conv3_2_conv conv3_2_b
    (conv3_2_out :: TBF)       = TFops.relu conv3_2_bias

  -- conv3_3
    (conv3_3_W :: TBF)         = TFops.constant (TF.Shape ([3,3,256,256])) ( Json.conv3_3_W weightDict)
    (conv3_3_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 )
                                 conv3_2_out conv3_3_W
    (conv3_3_b :: TBF)         = TFops.constant (TF.Shape ([256])) (Json.conv3_3_b weightDict)
    (conv3_3_bias :: TBF)      = TFops.add conv3_3_conv conv3_3_b
    (conv3_3_out :: TBF)       = TFops.relu conv3_3_bias

  -- pool3
    (pool3 :: TBF)             = TFops.avgPool'
                                 (
                                   (TFbuild.opAttr "ksize"   .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "strides" .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME"))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 )
                                 conv3_3_out
  -------------------------------------
-- block 4
  -------------------------------------
  -- conv4_1
    (conv4_1_W :: TBF)         = TFops.constant (TF.Shape ([3,3,256,512])) ( Json.conv4_1_W weightDict)
    (conv4_1_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 pool3 conv4_1_W
    (conv4_1_b :: TBF)         = TFops.constant (TF.Shape ([512])) (Json.conv4_1_b weightDict)
    (conv4_1_bias :: TBF)      = TFops.add conv4_1_conv conv4_1_b
    (conv4_1_out :: TBF)       = TFops.relu conv4_1_bias

  -- conv4_2
    (conv4_2_W :: TBF)         = TFops.constant (TF.Shape ([3,3,512,512])) ( Json.conv4_2_W weightDict)
    (conv4_2_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 conv4_1_out conv4_2_W
    (conv4_2_b :: TBF)         = TFops.constant (TF.Shape ([512])) (Json.conv4_2_b weightDict)
    (conv4_2_bias :: TBF)      = TFops.add conv4_2_conv conv4_2_b
    (conv4_2_out :: TBF)       = TFops.relu conv4_2_bias

  -- conv4_3
    (conv4_3_W :: TBF)         = TFops.constant (TF.Shape ([3,3,512,512])) ( Json.conv4_3_W weightDict)
    (conv4_3_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 conv4_2_out conv4_3_W
    (conv4_3_b :: TBF)         = TFops.constant (TF.Shape ([512])) (Json.conv4_3_b weightDict)
    (conv4_3_bias :: TBF)      = TFops.add conv4_3_conv conv4_3_b
    (conv4_3_out :: TBF)       = TFops.relu conv4_3_bias

  -- pool4
    (pool4 :: TBF)             = TFops.avgPool'
                                 (
                                   (TFbuild.opAttr "ksize"   .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "strides" .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME"))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 )
                                 conv4_3_out

  -------------------------------------
-- block 5
  -------------------------------------
  -- conv5_1
    (conv5_1_W :: TBF)         = TFops.constant (TF.Shape ([3,3,512,512])) ( Json.conv5_1_W weightDict)
    (conv5_1_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 pool4 conv5_1_W
    (conv5_1_b :: TBF)         = TFops.constant (TF.Shape ([512])) (Json.conv5_1_b weightDict)
    (conv5_1_bias :: TBF)      = TFops.add conv5_1_conv conv5_1_b
    (conv5_1_out :: TBF)       = TFops.relu conv5_1_bias

  -- conv5_2
    (conv5_2_W :: TBF)         = TFops.constant (TF.Shape ([3,3,512,512])) ( Json.conv5_2_W weightDict)
    (conv5_2_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 conv5_1_out conv5_2_W
    (conv5_2_b :: TBF)         = TFops.constant (TF.Shape ([512])) (Json.conv5_2_b weightDict)
    (conv5_2_bias :: TBF)      = TFops.add conv5_2_conv conv5_2_b
    (conv5_2_out :: TBF)       = TFops.relu conv5_2_bias

  -- conv5_3
    (conv5_3_W :: TBF)         = TFops.constant (TF.Shape ([3,3,512,512])) ( Json.conv5_3_W weightDict)
    (conv5_3_conv :: TBF)      = TFops.conv2D'
                                 (
                                   (TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME" ))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 . (TFbuild.opAttr "use_cudnn_on_gpu" .~ True)
                                 )
                                 conv5_2_out conv5_3_W
    (conv5_3_b :: TBF)         = TFops.constant (TF.Shape ([512])) (Json.conv5_3_b weightDict)
    (conv5_3_bias :: TBF)      = TFops.add conv5_3_conv conv5_3_b
    (conv5_3_out :: TBF)       = TFops.relu conv5_3_bias

  -- pool5
    (pool5 :: TBF)             = TFops.avgPool'
                                 (
                                   (TFbuild.opAttr "ksize"   .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "strides" .~ ([1,2,2,1] :: [I.Int64]))
                                 . (TFbuild.opAttr "padding" .~ (B.pack "SAME"))
                                 . (TFbuild.opAttr "data_format" .~ (B.pack "NHWC"))
                                 )
                                 conv5_3_out





--vgg16test :: Json.Weights -> TBF -> TF.Build (TBI32)
--vgg16test weightDict input = do
--  return $ TFops.shape  (conv5_1 $ vgg16 weightDict input)
--
--vgg16testsession :: Json.Weights -> [Float]  -> TF.Session (V.Vector I.Int32)
--vgg16testsession weightDict imgList = do
--  
--  let imgTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (imgList)
--
--  result <- TF.build (vgg16test weightDict imgTensor)
--  TF.run result
--
--vgg16testio :: IO()
--vgg16testio = do
--  weightDict <- Json.readWeights "/home/johannes/has/styletransfer/src/weights_flat.json"
--  imgList <- Img.listFromImage "/home/johannes/has/styletransfer/owl.jpeg"
--  result <- TFsess.runSession (vgg16testsession weightDict (concat $ concat $imgList))
--  print result
--  putStrLn "hallo"

