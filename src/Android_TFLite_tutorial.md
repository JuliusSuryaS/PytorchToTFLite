 # Android TFLite Tutorial
This is tutorial for using tflite flat buffers model in android. The official website is lack of details in using the tflite model, so I write a simple tutorial with little bit more details. The model is used from converted pytroch model, check [here]() for conversion tutorial.<br>

### Define the model
Running tflite model in android is fairly easy, first we need to define the `Interpreter` or our network model.
```Java
Interpreter net = new Interpreter(tflite_model);
```
In the current version, `Interpreter` class accepts only java `File` while other parameters is deprecated. For this tutorial we will use the `MappedByteBuffer` to load our model.

```Java
 private MappedByteBuffer loadTFLiteModel(String model_file) throws IOException {
        AssetFileDescriptor fileDescriptor = appContext.getAssets().openFd(model_file);
        FileInputStream fstream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fchannel = fstream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fchannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
```
This `loadTFLiteModel` takes model name in string format from the `asset` folder in our android studio. If we use `File` then we can simply pass the `File` to the constructor. Then to load the model we call the function
```Java
tflite_model = loadTFLiteModel("MODEL_NAME.tflite");
Interpreter net = new Interpreter(tflite_model);
```

### Running inference
We can do forward operations of our model by calling `run` methods of the `Interpreter` class. It takes both input and output as the arguments.
```Java
net.run(inp, out);
```
The input and output of the model can be an array with the same size as the defined input and output in the model. For example if we use input and output float with size of `[1, 100, 100 ,3]`, tensor with batch size 1 and 3 channel. We define them as TFLite model
```java
private float [][][][] inp = new float [1][100][100][3];
private float [][][][] out = new float [1][100][100][3];
```

We can also passed `ByteBuffer` to the input, instead of array and this is more efficient. This is done by converting the input into `ByteBuffer`. For this example we use `Bitmap` input image. First we define 1D array to contain all the pixel values of the input image.
```java
private int[] inImgArr = [img.getWidth() * img.getHeight()];
private ByteBuffer inImgBuff = ByteBuffer.allocateDirect(BATCH_SIZE * img.getWidth() * img.getHeight() * 
                                                         CHANNEL_SIZE * BYTES_PER_CHANNEL);
```
`BYTES_PER_CHANNEL` is obtained by dividing the size of the `float` with the size of `byte`
```Java
private final int BYTES_PER_CHANNEL = Float.SIZE / Byte.SIZE;
```
The function to convert `Bitmap` image to `ByteBuffer`. This process is done by first putting the pixel values into the `inImgArr` array to store all the pixel values. Then we perform loop on the correct index, and store them using `addPixelValue()` function.

```Java
private void bitmapToByteBuffer(Bitmap img){
        if (inImageBuff == null){
            return;
        }
        this.inImageBuff.rewind(); // clears data

        // Put image bitmap values into 1D intValues array
        img.getPixels(inImageArr, 0, img.getWidth(), 0, 0, img.getWidth(), img.getHeight());

        // Loop over the 1D array values of intValues, and store the RGB in byte buffer
        int pixel = 0;
        for (int i = 0; i < img.getWidth(); ++i){
            for (int j = 0; j < img.getHeight(); ++j){
                final int val = inImageArr[pixel++];
                // Process the bitmap value to RGB values, and store it to byte buffer
                addPixelValue(val);
            }
        }
    }

```
<br>

To handle the pixel value in `Bitmap` format the `addPixelValue()` function is utilized. Bitmap function stores the RGB pixel value in the single 32 bits hexadecimal format (ARGB), to get the `uint8` (8 bits) value of each channel, the value must be shifted accordingly and masked with `0xFF` (255). After we get the pixel for each channel, we normalize the input to `0 - 1` range.

```Java
private void addPixelValue(int val){
        // Put to R-G-B
        float r = (val >> 16) & 0xFF;
        float g = (val >> 8) & 0xFF;
        float b =  val & 0xFF;
        this.inImageBuff.putFloat(r/255); // R
        this.inImageBuff.putFloat(g/255); // G
        this.inImageBuff.putFloat(b/255); // B
    }
```

<br>
We have finished preparing the input for inference, now we define the output array and bitmap from the inference.

```Java
// TF format is NHWC
private float[][][][] outImgArr = new Float [BATCH_SIZE][img.getHeight()][img.getWidth()][CHANNEL_SIZE]
```

For the inference process, we convert the input array to `ByteBuffer` then simply call `run` method and we are finished. 

```Java
bitmapToByteBuffer(img);
net.run(this.inImageBuff, this.outImgArr); 
```

The remaining thing is to handle output array and convert it to `Bitmap` image. First we initialize the empty bitmap image.

```Java
private Bitmap outImage = Bitmap.createBitmap(img.getWidth(), img.getHeight(), Bitmap.Config.ARGB_8888);
```

Next we convert the input from float array to `Bitmap` image using `floatArr2Bitmap()`. We loop over the index an get pixel on each RGB channel. The float value is normalized back to `uint8` and put it to the image.

```Java
private void floatArr2Bitmap (float[][][][] outImgArr){
        for (int i = 0; i < img.getWidth(); i++){
            for (int j = 0; j < img.getHeight(); j++){
                int red = (int)(outImgArr[0][j][i][0] * 255);
                int green = (int)(outImgArr[0][j][i][1] * 255);
                int blue = (int)(outImgArr[0][j][i][2] * 255);
                this.outImage.setPixel(i, j, Color.rgb(red, green, blue));
            }
        }
    }
```

After we finished with the inference process, we close the `Interpreter` to avoid memory leak by calling `net.close()`.
