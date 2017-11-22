--someFunc :: IO ()
--someFunc = putStrLn "someFunc"
--
--readPb :: Message GraphDef => FilePath -> IO GraphDef
--readPb path = do
--  pb <- BSL.readFile path
--  return $ decodeMessageOrDie $ BSL.toStrict pb
--
--pb :: Message GraphDef => IO GraphDef
--pb = readPb ("/home/johannes/has/styletransfer/src/vgg_16_two_inp_two_out.pb" :: FilePath)
--
----vgg :: String  -> TBF
--vgg :: (TBF, TBF)
--vgg  = (TF.tensorFromName (T.pack "strided_sladasdice") :: TBF, TF.tensorFromName (T.pack "stridasdased_slice_1") :: TBF)



{-
msetest :: Build (TBF)
msetest = do
  let a = constant (Shape ([1,224,224,3]))
-}


--batchFlatten :: TBF -> TBF
--batchFlatten inp = TFops.reshape (transposed_tensor) (new_shape)
--  where
--    (transposed_tensor :: TBF) = TFops.transpose (squeezed) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)
--    --squeezed = TFops.reshape inp (TFops.gather (TFops.shape inp) (TFops.constant (TF.Shape ([3] :: [I.Int64])) ([1,2,3]:: [I.Int64]) ) :: TBI32)
--    (squeezed :: TBF) = TFops.gather inp (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int64]))
--    new_shape = TFops.concat (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int32]))
--      ([
--      (TFops.cast (TFops.gather (TFops.shape squeezed) (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int64])))),
--      (TFops.cast (TFops.gather (TFops.shape squeezed) (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([1] :: [I.Int64]))))
--      ] :: [TBI32])


--batchtest :: TF.Build (TBI32)
--batchtest = do
--  return $ TFops.shape $  batchFlatten (TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) [1..150528] )
--
--batchtestSession :: TF.Session (V.Vector I.Int32)
--batchtestSession = do
--  result <- TF.build ( batchtest)
--  TF.run result


--gramtest :: TF.Build (TBI32)
--gramtest = do
--  return $ TFops.shape $ gramMatrix (TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) [1..150528] )

--gramSession :: TF.Session (V.Vector I.Int32)
--gramSession = do
--  result <- TF.build ( gramtest)
--  TF.run result







--loss ::TBF -> TBF -> TBF -> TBF -> TBF -> TBF -> TBF -> TBF
--loss imgTensorConv5 randTensorConv5 styleTensorConv2 randTensorConv2 styleTensorConv2 randTensorConv2 styleTensorConv2 randTensorConv2 = ((contentloss imgTensorConv5 randTensorConv5) `TFops.div` 10) `TFops.add`
--  ((styleloss1 styleTensorConv1 randTensorConv1) `TFops.add` (styleloss2 styleTensorConv2 randTensorConv2) `TFops.add` (styleloss3 styleTensorConv3 randTensorConv3))

--graph :: Json.Weights ->  V.Vector Float -> V.Vector Float -> V.Vector Float -> TF.Build (TBF)
--graph weights imgVector styleVector randVector = do
-- let imgTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList imgVector)
--     styleTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList styleVector)
--     randTensor = TFops.constant (TF.Shape ([1,224,224,3] :: [I.Int64])) (V.toList randVector)
--     imgTensorConv = Vgg16.vgg16.vgg16.vgg16.vgg16 weights imgTensor
--     randTensorConv =Vgg16.vgg16weights randTensor
-- return $ loss imgTensorConv randTensorConv styleTensor randTensor













batchFlatten0 :: TBF -> TBF
batchFlatten0 inp = TFops.reshape (TFops.transpose squeezed ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [-1 ,224 * 224]) :: TBI32)
  where (squeezed :: TBF) = TFops.gather inp (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int64]))

batchFlatten1 :: TBF -> TBF
batchFlatten1 inp = TFops.reshape (TFops.transpose (squeezed) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [-1 ,224 * 224]) :: TBI32)
  where (squeezed :: TBF) = TFops.gather inp (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int32]))

batchFlatten2 :: TBF -> TBF
batchFlatten2 inp = TFops.reshape (TFops.transpose (squeezed) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [-1 ,112 * 112]) :: TBI32)
  where (squeezed :: TBF) = TFops.gather inp (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int32]))

batchFlatten3 :: TBF -> TBF
batchFlatten3 inp = TFops.reshape (TFops.transpose (squeezed) ((TFops.constant (TF.Shape ([3] :: [I.Int64])) [2,0,1]) :: TBI32)) ((TFops.constant (TF.Shape ([2] :: [I.Int64])) [-1 ,56 * 56]) :: TBI32)
  where (squeezed :: TBF) = TFops.gather inp (TFops.constant (TF.Shape ([1] :: [I.Int64])) ([0] :: [I.Int32]))

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


styleloss0 :: TBF -> TBF -> TBF
styleloss0 styleTensorConv randTensorConv = mse (gramMatrix0 styleTensorConv) (gramMatrix0 randTensorConv)

styleloss1 :: TBF -> TBF -> TBF
styleloss1 styleTensorConv randTensorConv = mse (gramMatrix1 styleTensorConv) (gramMatrix1 randTensorConv)

styleloss2 :: TBF -> TBF -> TBF
styleloss2 styleTensorConv randTensorConv = mse (gramMatrix2 styleTensorConv) (gramMatrix2 randTensorConv)

styleloss3 :: TBF -> TBF -> TBF
styleloss3 styleTensorConv randTensorConv = mse (gramMatrix3 styleTensorConv) (gramMatrix3 randTensorConv)

































