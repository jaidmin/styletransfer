{-# LANGUAGE DeriveGeneric, OverloadedStrings #-}

--exporting all the relevant functions and data constructors
module JsonHandling
    ( Weights (..),
      getJSON
    , readWeights
    ) where

import Data.Aeson
import GHC.Generics
import qualified Data.ByteString.Lazy as B

--read a Json file from a specified filePath
getJSON :: FilePath -> IO B.ByteString
getJSON = B.readFile 

--
readWeights :: FilePath -> IO Weights
readWeights = fmap maybeToVal . fmap decode . getJSON

maybeToVal :: Maybe a -> a
maybeToVal (Just x) = x

data Weights = 
  Weights {
          conv1_1_W :: [Float]
        , conv1_2_W :: [Float]
        , conv2_1_W :: [Float]
        , conv2_2_W :: [Float]
        , conv3_1_W :: [Float]
        , conv3_2_W :: [Float]
        , conv3_3_W :: [Float]
        , conv4_1_W :: [Float]
        , conv4_2_W :: [Float]
        , conv4_3_W :: [Float]
        , conv5_1_W :: [Float]
        , conv5_2_W :: [Float]
        , conv5_3_W :: [Float]
        , conv1_1_b :: [Float]
        , conv1_2_b :: [Float]
        , conv2_1_b :: [Float]
        , conv2_2_b :: [Float]
        , conv3_1_b :: [Float]
        , conv3_2_b :: [Float]
        , conv3_3_b :: [Float]
        , conv4_1_b :: [Float]
        , conv4_2_b :: [Float]
        , conv4_3_b :: [Float]
        , conv5_1_b :: [Float]
        , conv5_2_b :: [Float]
        , conv5_3_b :: [Float]
        } deriving (Show, Generic)

instance FromJSON Weights
instance ToJSON Weights
