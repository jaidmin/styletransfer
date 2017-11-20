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
import qualified Data.Vector as V
import qualified Data.Int as I
import qualified Data.Text as T
import System.Environment (getArgs)
import qualified System.Random as R
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Char8 as B

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
import qualified TensorFlow.GenOps.Core as TFops (conv2D,conv2D', maxPool, biasAdd, square, squeeze, onesLike)
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
--vgg  outnode = tensorFromName (T.pack outnode)

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

-- not good becauz hard coded tensor shape
batchFlatten :: TBF -> TBF
batchFlatten inp = TFops.reshape (TFops.transpose (TFops.squeeze inp) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [3 ,224 * 224]) :: TBI32)

batchtest :: TF.Build (TBI32)
batchtest = do
  return $ TFops.shape $  batchFlatten (TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) [1..150528] )

batchtestSession :: TF.Session (V.Vector I.Int32)
batchtestSession = do
  result <- TF.build ( batchtest)
  TF.run result

gramMatrix :: TBF -> TBF
gramMatrix inp = inp_flat `TFops.matMul` (TFops.transpose inp_flat ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [1,0]) :: TBI32))
  where inp_flat = batchFlatten inp

gramtest :: TF.Build (TBI32)
gramtest = do
  return $ TFops.shape $ gramMatrix (TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) [1..150528] )

gramSession :: TF.Session (V.Vector I.Int32)
gramSession = do
  result <- TF.build ( gramtest)
  TF.run result

styleloss :: TBF -> TBF -> TBF
styleloss styleTensor randTensor = mse (gramMatrix styleTensor) (gramMatrix randTensor)

loss :: TBF -> TBF -> TBF -> TBF -> TBF
loss imgTensorConv randTensorConv styleTensor randTensor = (contentloss imgTensorConv randTensorConv) `TFops.add` (styleloss styleTensor randTensor)

update :: Float -> TBF -> [TFvar.Variable Float] -> TF.Build (TF.ControlNode)
update lr loss vars = TFmin.minimizeWith (TFmin.gradientDescent lr) loss vars

graph :: V.Vector Float -> V.Vector Float -> V.Vector Float -> TF.Build (TBF)
graph imgVector styleVector randVector = do
  let (imgTensorConv, randTensorConv) = vgg
  let imgTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList imgVector)
  let styleTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList styleVector)
  let randTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList randVector)
  return $ loss imgTensorConv randTensorConv styleTensor randTensor


-- maybe insert runsession in front of "do"
session :: GraphDef -> V.Vector Float -> V.Vector Float -> V.Vector Float -> TF.Session (V.Vector Float)
session graphdef imgVector styleVector randVector = do
  TFsess.addGraphDef graphdef
  let contentPlaceholder = TF.tensorFromName (T.pack "PlWaceholder")  :: TBF
  let randPlaceholder    = TF.tensorFromName (T.pack "Placeholder_1") :: TBF

  renderedContent <- TF.render contentPlaceholder
  renderedRandom  <- TF.render randPlaceholder


  let feededImg  = TF.feed renderedContent (TF.encodeTensorData (TF.Shape ([1,224,224,3] :: [I.Int64])) imgVector)
  let feededRand = TF.feed renderedRandom (TF.encodeTensorData (TF.Shape ([1,224,224,3] :: [I.Int64])) randVector)


  loss <- TF.build (graph imgVector styleVector randVector)

  grads <- TFgrad.gradients contentPlaceholder [renderedRandom, renderedContent]

  --runWithFeeds [feededRand, feededImg] (Prelude.head grads) :: Session (Vector Float)
  TF.run (Prelude.head grads) :: TF.Session (V.Vector Float)



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

  [imgPath, stylePath, savePath] <- getArgs
  imgList <- Img.listFromImage imgPath
  styleList <- Img.listFromImage stylePath

  let imgVec = V.fromList $ Prelude.concat $ Prelude.concat imgList :: V.Vector Float
  let styleVec = V.fromList $ Prelude.concat $ Prelude.concat styleList :: V.Vector Float

  -- read graphdef from file
  gd <- pb

  --generate random vector
  let randVec = V.fromList $ (Prelude.take 150528 (R.randoms (R.mkStdGen seed)) :: [Float]) :: V.Vector Float

  --create and save output
  loss <- TF.runSession (session gd imgVec styleVec randVec)
  --imageFromList (toList outVec1) (savePath Prelude.++ "1.jpeg")
  --imageFromList (toList outVec2) (savePath Prelude.++ "2.jpeg")
  --print (outVec1 == outVec1)
  print loss
  putStrLn "hallo"

flatten4_1 :: [[[[Float]]]] -> [Float]
flatten4_1 inp = concat $ concat $ concat inp

vgg16 :: Json.Weights -> TBF -> TBF
vgg16 weightDict input = 
  let (mean :: TBF) = TFops.constant (TF.Shape ([1,1,1,3] :: [I.Int64])) [123.68, 116.779, 103.939] 
      normalized = input `TFops.sub` mean
      conv1 = TFops.conv2D' ((TFbuild.opAttr "strides" .~ ([1,1,1,1] :: [I.Int64]))
                               . (TFbuild.opAttr "padding" .~ (B.pack "VALID" )  ))
              normalized (TFops.constant (TF.Shape ([3,3,3,64])) (flatten4_1 $ Json.conv1_1_W weightDict))
  in
    conv1


vgg16test :: Json.Weights -> TBF -> TF.Build (TBI32)
vgg16test weightDict input = do
  return $ TFops.shape (vgg16 weightDict input)

vgg16testsession :: Json.Weights -> [Float]  -> TF.Session (V.Vector I.Int32)
vgg16testsession weightDict imgList = do
  
  let imgTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (imgList)

  result <- TF.build (vgg16test weightDict imgTensor)
  TF.run result

vgg16testio :: IO()
vgg16testio = do
  weightDict <- Json.readWeights "/home/johannes/has/styletransfer/src/weights.json"
  imgList <- Img.listFromImage "/home/johannes/has/styletransfer/owl.jpeg"
  result <- TFsess.runSession (vgg16testsession weightDict (concat $ concat $imgList))
  print result
  putStrLn "hallo"

