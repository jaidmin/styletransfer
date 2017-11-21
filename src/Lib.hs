{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib
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

-- (old) protobuf imports
import Data.ProtoLens (Message, decodeMessageOrDie)
import qualified  Data.ByteString.Lazy as BSL (toStrict, readFile) 
import Data.ProtoLens.TextFormat
import Proto.Tensorflow.Core.Framework.Graph

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

someFunc :: IO ()
someFunc = putStrLn "someFunc"

readPb :: Message GraphDef => FilePath -> IO GraphDef
readPb path = do
  pb <- BSL.readFile path
  return $ decodeMessageOrDie $ BSL.toStrict pb

pb :: Message GraphDef => IO GraphDef
pb = readPb ("/home/johannes/has/styletransfer/src/vgg_16_two_inp_two_out.pb" :: FilePath)

--vgg :: String  -> TBF
vgg :: (TBF, TBF)
vgg  = (TF.tensorFromName (T.pack "strided_sladasdice") :: TBF, TF.tensorFromName (T.pack "stridasdased_slice_1") :: TBF)


mse :: TBF -> TBF -> TBF
mse pre targ = TFops.reduceMean $ TFops.square (pre `TFops.sub` targ)

{-
msetest :: Build (TBF)
msetest = do
  let a = constant (Shape ([1,224,224,3]))
-}

contentloss :: TBF -> TBF -> TBF
contentloss imgTensorConv randTensorConv = mse imgTensorConv randTensorConv

--gets block 2 (112,112,128)
-- not good becauz hard coded tensor shape
batchFlatten0 :: TBF -> TBF
batchFlatten0 inp = TFops.reshape (TFops.transpose (ourSqueeze inp) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [-1 ,224 * 224]) :: TBI32)
  where ourSqueeze inp = TFops.reshape inp (TFops.constant (TF.Shape ([3] :: [I.Int64])) ([224, 224, 3] :: [I.Int32]))


batchFlatten :: TBF -> TBF
batchFlatten inp = TFops.reshape (transposed_tensor) (new_shape)
  where
    transposed_tensor = TFops.transpose (squeezed) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)
    --squeezed = TFops.reshape inp (TFops.gather (TFops.shape inp) (TFops.constant (TF.Shape ([3] :: [I.Int64])) ([1,2,3]:: [I.Int64]) ) :: TBI32)
    squeezed = TFops.gather inp (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int64]))
    (new_shape :: TBI32) = TFops.concat 0 [(TFops.gather (TFops.shape transposed_tensor) (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int64]))) , (TFops.gather (TFops.shape transposed_tensor) (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([1] :: [I.Int64])))]


batchFlatten1 :: TBF -> TBF
batchFlatten1 inp = TFops.reshape (TFops.transpose (ourSqueeze inp) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [-1 ,224 * 224]) :: TBI32)
  where ourSqueeze inp = TFops.reshape inp (TFops.constant (TF.Shape ([3] :: [I.Int64])) ([224, 224, 64] :: [I.Int32]))
batchFlatten2 :: TBF -> TBF
batchFlatten2 inp = TFops.reshape (TFops.transpose (ourSqueeze inp) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [-1 ,112 * 112]) :: TBI32)
  where ourSqueeze inp = TFops.reshape inp (TFops.constant (TF.Shape ([3] :: [I.Int64])) ([112,112,128] :: [I.Int32]))

batchFlatten3 :: TBF -> TBF
batchFlatten3 inp = TFops.reshape (TFops.transpose (ourSqueeze inp) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [-1 ,56 * 56]) :: TBI32)
  where ourSqueeze inp = TFops.reshape inp (TFops.constant (TF.Shape ([3] :: [I.Int64] )) ([56, 56, 256] :: [I.Int32]))




--batchtest :: TF.Build (TBI32)
--batchtest = do
--  return $ TFops.shape $  batchFlatten (TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) [1..150528] )
--
--batchtestSession :: TF.Session (V.Vector I.Int32)
--batchtestSession = do
--  result <- TF.build ( batchtest)
--  TF.run result
gramMatrix :: TBF -> TBF
gramMatrix inp = (inp_flat `TFops.matMul` (TFops.transpose inp_flat ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [1,0]) :: TBI32))) `TFops.div`   division_factor
  where
    inp_flat = batchFlatten0 inp
    (division_factor :: TBF) = TFops.cast $ ((TFops.gather (TFops.shape inp_flat) (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([1] :: [I.Int64])))) `TFops.mul` ((TFops.gather (TFops.shape inp_flat) (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int64]))))


gramMatrix0 :: TBF -> TBF
gramMatrix0 inp = (inp_flat `TFops.matMul` (TFops.transpose inp_flat ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [1,0]) :: TBI32))) `TFops.div`   (224 * 224 * 3)
  where inp_flat = batchFlatten0 inp


gramMatrix1 :: TBF -> TBF
gramMatrix1 inp = (inp_flat `TFops.matMul` (TFops.transpose inp_flat ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [1,0]) :: TBI32))) `TFops.div`   (224 * 224 * 64)
  where inp_flat = batchFlatten1 inp

gramMatrix2 :: TBF -> TBF
gramMatrix2 inp = (inp_flat `TFops.matMul` (TFops.transpose inp_flat ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [1,0]) :: TBI32))) `TFops.div`   (112 * 112 * 128)
  where inp_flat = batchFlatten2 inp

gramMatrix3 :: TBF -> TBF
gramMatrix3 inp = (inp_flat `TFops.matMul` (TFops.transpose inp_flat ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [1,0]) :: TBI32))) `TFops.div`   (56 * 56 * 256)
  where inp_flat = batchFlatten3 inp


--gramtest :: TF.Build (TBI32)
--gramtest = do
--  return $ TFops.shape $ gramMatrix (TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) [1..150528] )

--gramSession :: TF.Session (V.Vector I.Int32)
--gramSession = do
--  result <- TF.build ( gramtest)
--  TF.run result
styleloss :: TBF -> TBF -> TBF
styleloss styleTensorConv randTensorConv = mse (gramMatrix styleTensorConv) (gramMatrix randTensorConv)


styleloss0 :: TBF -> TBF -> TBF
styleloss0 styleTensorConv randTensorConv = mse (gramMatrix0 styleTensorConv) (gramMatrix0 randTensorConv)

styleloss1 :: TBF -> TBF -> TBF
styleloss1 styleTensorConv randTensorConv = mse (gramMatrix1 styleTensorConv) (gramMatrix1 randTensorConv)

styleloss2 :: TBF -> TBF -> TBF
styleloss2 styleTensorConv randTensorConv = mse (gramMatrix2 styleTensorConv) (gramMatrix2 randTensorConv)

styleloss3 :: TBF -> TBF -> TBF
styleloss3 styleTensorConv randTensorConv = mse (gramMatrix3 styleTensorConv) (gramMatrix3 randTensorConv)

--loss ::TBF -> TBF -> TBF -> TBF -> TBF -> TBF -> TBF -> TBF
--loss imgTensorConv5 randTensorConv5 styleTensorConv2 randTensorConv2 styleTensorConv2 randTensorConv2 styleTensorConv2 randTensorConv2 = ((contentloss imgTensorConv5 randTensorConv5) `TFops.div` 10) `TFops.add`
--  ((styleloss1 styleTensorConv1 randTensorConv1) `TFops.add` (styleloss2 styleTensorConv2 randTensorConv2) `TFops.add` (styleloss3 styleTensorConv3 randTensorConv3))


update :: Float -> TBF -> [TFvar.Variable Float] -> TF.Build (TF.ControlNode)
update lr loss vars = TFmin.minimizeWith (TFmin.gradientDescent lr) loss vars

--graph :: Json.Weights ->  V.Vector Float -> V.Vector Float -> V.Vector Float -> TF.Build (TBF)
--graph weights imgVector styleVector randVector = do
-- let imgTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList imgVector)
--     styleTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList styleVector)
--     randTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList randVector)
--     imgTensorConv = vgg16 weights imgTensor
--     randTensorConv = vgg16 weights randTensor
-- return $ loss imgTensorConv randTensorConv styleTensor randTensor


-- maybe insert runsession in front of "do"
session :: Json.Weights -> V.Vector Float -> V.Vector Float -> Int -> TFsess.Session ( V.Vector Float)
session weights imgVector styleVector steps = do

  let
    seed = 123456
   -- randList = (Prelude.take 150528 (R.randomRs (0,255) (R.mkStdGen seed)) :: [Float])
    randList = [122.5 | _ <- [1..150528]]

 -- Trace.traceM ("randList: " ++ (show randList))
  (randVar :: TFvar.Variable Float) <- TFvar.initializedVariable (TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64]))  randList )

  let
      imgTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList imgVector)
      styleTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList styleVector)
      randTensor   = (TFvar.readValue randVar)

      imgTensorConv3 = conv3_1 $ vgg16 weights imgTensor
      imgTensorConv4 = conv4_1 $ vgg16 weights imgTensor
      imgTensorConv5 = conv5_1 $ vgg16 weights imgTensor

      randTensorConv1 = conv1_1 $ vgg16 weights randTensor
      styleTensorConv1 = conv1_1 $ vgg16 weights styleTensor
      randTensorConv2 = conv2_1 $ vgg16 weights randTensor
      styleTensorConv2 = conv2_1 $ vgg16 weights styleTensor
      randTensorConv3 = conv3_1 $ vgg16 weights randTensor
      styleTensorConv3 = conv3_1 $ vgg16 weights styleTensor
      
      randTensorConv4 = conv4_1 $ vgg16 weights randTensor
      randTensorConv5 = conv5_1 $ vgg16 weights randTensor

      contentLoss = (contentloss imgTensorConv4 randTensorConv4) `TFops.div` 100


      --styleLoss = ((styleloss1 styleTensorConv1 randTensorConv1) `TFops.add`  (styleloss2 styleTensorConv2 randTensorConv2) `TFops.add` (styleloss3 styleTensorConv3 randTensorConv3) ) `TFops.div` 3
      styleLoss = (styleloss styleTensor randTensor)
  
      thisloss = contentLoss `TFops.add`  styleLoss

  let admcfg = TFmin.AdamConfig 0.1 0.9 0.999 1e-8
  --update <- TFmin.minimizeWith (TFmin.gradientDescent 0.1) thisloss [randVar]
  updateStyle <- TFmin.minimizeWith (TFmin.adam' admcfg) styleLoss [randVar]
  let trainstepStyle = TF.run updateStyle

  forM_ ([0.. (steps `div` 2)] :: [Int]) $ \i -> do
    trainstepStyle
    when  (i `mod` 100 == 0) $ do
      (styleLossRightNow :: V.Vector Float) <- TF.run styleLoss
      (contentLossRightNow :: V.Vector Float) <- TF.run contentLoss
      liftIO $ putStrLn $ "step: " ++ (show i) ++ " styleloss: " ++ (show styleLossRightNow) ++ " contentloss: " ++ (show contentLossRightNow)

  

  updateAll <- TFmin.minimizeWith (TFmin.adam' admcfg) thisloss [randVar]
  let trainstepAll = TF.run updateAll
  forM_ ([0..(steps `div` 2)] :: [Int]) $ \i -> do
    trainstepAll
    when  (i `mod` 100 == 0) $ do
      (styleLossRightNow :: V.Vector Float) <- TF.run styleLoss
      (contentLossRightNow :: V.Vector Float) <- TF.run contentLoss
      liftIO $ putStrLn $ "step: " ++ (show i) ++ " styleloss: " ++ (show styleLossRightNow) ++ " contentloss: " ++ (show contentLossRightNow)
  let output = (TFvar.readValue randVar)
  TF.run output
  



--test :: Session (Vector Float)
--test = runSession $ do
--  let tensor = TF.zeros (Shape ([1,3,224,224] :: [Int64]))
--  let tensor_flat = batchFlatten tensor
--  new_tensor <- build (graph tensor_flat)
--  run new_tensor



simpleFunc :: TBF -> TF.Build (TBF)
simpleFunc tensor = do
  let new_tensor =  (tensor `TFops.sub` tensor)
  return new_tensor

--test :: Session (Vector Float)
--test = runSession $ do
--  let tensor = TF.zeros (Shape ([3,224,224] :: [Int64]))
--  new_tensor <- build (simpleFunc tensor)
--  run new_tensor



mainLib :: IO ()
mainLib = do
  --random seed
  let seed = 14783

  [imgPath, stylePath, savePath, steps] <- getArgs
  imgList <- Img.listFromImage imgPath
  styleList <- Img.listFromImage stylePath
  let (stepsInt :: Int) =round $ read steps

  let imgVec = V.fromList $ Prelude.concat $ Prelude.concat imgList :: V.Vector Float
  let styleVec = V.fromList $ Prelude.concat $ Prelude.concat styleList :: V.Vector Float

  -- read weights from json
  Trace.traceIO "Starting to load the weighs!"
  weights <- Json.readWeights "/home/johannes/has/styletransfer/src/weights_flat.json"
  Trace.traceIO "Weights loaded! "
  --create and save output

  output <- TF.runSession (session weights imgVec styleVec stepsInt)
 -- let processed_output = map (*255) (V.toList output)
  let processed_output = V.toList output
  Img.imageFromList processed_output (savePath Prelude.++ ".jpeg")
  --imageFromList (toList outVec2) (savePath Prelude.++ "2.jpeg")

  putStrLn "executed"

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





vgg16test :: Json.Weights -> TBF -> TF.Build (TBI32)
vgg16test weightDict input = do
  return $ TFops.shape  (conv5_1 $ vgg16 weightDict input)

vgg16testsession :: Json.Weights -> [Float]  -> TF.Session (V.Vector I.Int32)
vgg16testsession weightDict imgList = do
  
  let imgTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (imgList)

  result <- TF.build (vgg16test weightDict imgTensor)
  TF.run result

vgg16testio :: IO()
vgg16testio = do
  weightDict <- Json.readWeights "/home/johannes/has/styletransfer/src/weights_flat.json"
  imgList <- Img.listFromImage "/home/johannes/has/styletransfer/owl.jpeg"
  result <- TFsess.runSession (vgg16testsession weightDict (concat $ concat $imgList))
  print result
  putStrLn "hallo"

