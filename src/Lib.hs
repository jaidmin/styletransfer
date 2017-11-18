{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Lib
    ( someFunc
    ) where
import TensorFlow.Ops hiding (initializedVariable, initializedVariable')
import Prelude hiding (readFile)
import TensorFlow.Build
import TensorFlow.Core
import Data.ProtoLens (Message, decodeMessageOrDie)
import Data.ByteString.Lazy (toStrict, readFile)
import Data.ProtoLens.TextFormat
import Proto.Tensorflow.Core.Framework.Graph
import Data.Vector
import Data.Int
import TensorFlow.Variable 

someFunc :: IO ()
someFunc = putStrLn "someFunc"

readPb :: Message GraphDef => FilePath -> IO GraphDef
readPb path = do
  pb <- readFile path
  return $ decodeMessageOrDie $ toStrict pb

pb :: Message GraphDef => IO GraphDef
pb = readPb ("/home/johannes/has/styletransfer/src/vgg16_final.pb" :: FilePath)


vgg :: [Float] -> GraphDef -> Build (Tensor Build Float)
vgg list graphdef = do
  (inp :: Tensor Value Float) <- placeholder (Shape ([1,224,224,3]:: [Int64]))
  (var :: Variable Float) <- initializedVariable inp
  addGraphDef graphdef
  output = tensorFromName ""
  return
