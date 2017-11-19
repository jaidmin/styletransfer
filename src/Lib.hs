{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib
     where
import TensorFlow.Ops as TF --hiding (initializedVariable, initializedVariable') 
import Prelude hiding (readFile)
import TensorFlow.Build
import TensorFlow.Core as TF
import Data.ProtoLens (Message, decodeMessageOrDie)
import Data.ByteString.Lazy (toStrict, readFile)
import Data.ProtoLens.TextFormat
import Proto.Tensorflow.Core.Framework.Graph
import Data.Vector as V
import Data.Int
import TensorFlow.Tensor as TF
import TensorFlow.GenOps.Core as TF (square, squeeze, onesLike)
import TensorFlow.Minimize as TF
import TensorFlow.Variable 
import qualified Data.Text as T
import TensorFlow.Session (runWithFeeds)
import System.Environment (getArgs)
import ImageHandling
import System.Random

type TBF = Tensor Build Float
type TBI64= Tensor Build Int64
type TBI32= Tensor Build Int32

someFunc :: IO ()
someFunc = putStrLn "someFunc"

readPb :: Message GraphDef => FilePath -> IO GraphDef
readPb path = do
  pb <- readFile path
  return $ decodeMessageOrDie $ toStrict pb

pb :: Message GraphDef => IO GraphDef
--pb = readPb ("/home/johannes/has/styletransfer/src/optimized_inception_graph.pb" :: FilePath)
pb = readPb ("/home/johannes/has/styletransfer/src/vgg_16_frozen_graph.pb" :: FilePath)

--vgg :: String  -> TBF
--vgg  outnode = tensorFromName (T.pack outnode)

vgg :: Tensor Ref Float -> String -> Build(Tensor Build Float)
vgg inp outnode = do
  let input = tensorRefFromName (T.pack "Placeholder") :: Tensor Ref Float
  new_input <- TF.assign input inp 
  return $ tensorFromName (T.pack outnode) ::Build( Tensor Build Float)


mse :: TBF -> TBF -> TBF
mse pre targ = TF.reduceMean $ TF.square (pre `TF.sub` targ)

contentloss :: TBF -> TBF -> TBF
contentloss imgTensorConv randTensorConv = mse imgTensorConv randTensorConv

-- not good becauz hard coded tensor shape
batchFlatten :: TBF -> TBF
batchFlatten inp = TF.reshape (TF.transpose (TF.squeeze inp) ((constant (Shape ([3] :: [Int64])) [2,0,1]) :: TBI32)) ((constant (Shape ([2] :: [Int64])) [3 ,224 * 224]) :: TBI32)

gramMatrix :: TBF -> TBF
gramMatrix inp = inp_flat `TF.matMul` (TF.transpose inp_flat ((constant (Shape ([3] :: [Int64])) [2,0,1]) :: TBI32))
  where inp_flat = batchFlatten inp

styleloss :: TBF -> TBF -> TBF
styleloss styleTensor randTensor = mse (gramMatrix $ batchFlatten styleTensor) (gramMatrix $ batchFlatten randTensor)

loss :: TBF -> TBF -> TBF -> TBF -> TBF
loss imgTensorConv randTensorConv styleTensor randTensor = (contentloss imgTensorConv randTensorConv) `add` (styleloss styleTensor randTensor)

update :: Float -> TBF -> [Variable Float] -> Build (ControlNode)
update lr loss vars = TF.minimizeWith (TF.gradientDescent lr) loss vars
{- graph :: String -> TBF -> TBF -> TBF -> Build (TBF)
graph outnode imgTensor styleTensor randTensor = do
  imgTensorConv <- (vgg outnode)
  randTensorConv <- (vgg outnode )
  return $ loss imgTensorConv randTensorConv styleTensor randTensor
-}

-- maybe insert runsession in front of "do"
session :: GraphDef -> String -> Vector Float -> Vector Float -> Vector Float -> Session (Vector Float, Vector Float) 
session graphdef outnode imgVector styleVector randVector = do
  addGraphDef graphdef
  randTensorVar <-  (TF.initializedVariable (constant (Shape ([3,224,224] :: [Int64])) (V.toList randVector)))
  --let feededImg =  TF.feed input (encodeTensorData (Shape ([1,224,224,3] :: [Int64])) imgVector)
  --let feededRand =  TF.feed input (encodeTensorData (Shape ([1,224,224,3] :: [Int64])) randVector)

  --let imgTensor = TF.variable (Shape ([3,224,224] :: [Int64])) (toList imgVector) :: Tensor Ref Float

  --vgg1out <- build (vgg imgTensor outnode)
  vgg2out <- build (vgg (randTensorVar) outnode)

  --run1 <- run vgg1out :: Session (Vector Float)
  run2 <-  run vgg2out :: Session (Vector Float)

  return (run2, run2)













  
--test :: Session (Vector Float)
--test = runSession $ do
--  let tensor = TF.zeros (Shape ([1,3,224,224] :: [Int64])) 
--  let tensor_flat = batchFlatten tensor
--  new_tensor <- build (graph tensor_flat)
--  run new_tensor



simpleFunc :: TBF -> Build (TBF)
simpleFunc tensor = do
  let new_tensor =  (tensor `TF.sub` tensor)
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
  imgList <- listFromImage imgPath
  styleList <- listFromImage stylePath

  let imgVec = fromList $ Prelude.concat $ Prelude.concat imgList :: Vector Float
  let styleVec = fromList $ Prelude.concat $ Prelude.concat styleList :: Vector Float

  -- read graphdef from file
  gd <- pb

  --generate random vector
  let randVec = fromList $ (Prelude.take 150528 (randoms (mkStdGen seed)) :: [Float]) :: Vector Float

  --create and save output 
  (outVec1, outVec2) <- TF.runSession (session gd "vgg16/block5_pool/MaxPool" imgVec styleVec randVec)
  imageFromList (toList outVec1) (savePath Prelude.++ "1.jpeg")
  imageFromList (toList outVec2) (savePath Prelude.++ "2.jpeg")
  print (outVec1 == outVec1)
  putStrLn "hallo"





 
 
