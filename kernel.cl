struct Dims {
  float xmin;
  float xmax;
  float ymin;
  float ymax;
  int width;
  int height;
  float zoom;
};
void HSLToRGB(int h, float s, float l, __global uchar *RGB, int id);
float HueToRGB(float v1, float v2, float vH);
__kernel void mandlebrot(__global uchar *RGB, __global struct Dims *dims) {
  int id = get_global_id(0);
  int w = dims->width;
  int h = dims->height;
  float fid = (float)id;
  int w2 = w * w;
  float col = (float)(id % w);
  int max_iter = 1000;
    float p_x = (col) / ((float)h) + (dims->xmin);
    float p_y = ((float)fid) / ((float)w2) + (dims->ymin);
    float x = 0.0;
    float y = 0.0;
    float next_x = 0.0;
    float next_y = 0.0;
    int i = 0;

    for (; i < max_iter; i += 1) {
      next_x = x * x - y * y + p_x;
      next_y = x * y + x * y + p_y;
      if (next_x * next_x + next_y * next_y > 1<<10){
        break;
      }
      x = next_x;
      y = next_y;
    }
  float val = (float)(i % max_iter);
    if (i < max_iter){
      float log_zn = log(sqrt(x*x + y*y));
      float nu =  log2(log_zn / log((float)max_iter));
      val = i - nu;
    }
  // little endian // BGR
  float val_hue = (float)(val / (float)max_iter);
  float offset = 220.0f;
  float hue = val_hue == 0.0f ? 0.0f :  fmod((pown(val_hue * (360 -offset ), 2)/sqrt(val_hue * (360 - offset)) + offset )   , 360.0f);
  HSLToRGB((int)hue, 1.0f,  sqrt(val/(float)max_iter), RGB, id);
}
void HSLToRGB(int h, float s, float l, __global uchar *RGB, int id) {
  unsigned char r = 0;
  unsigned char g = 0;
  unsigned char b = 0;

  if (s == 0) {
    RGB[3 * id + 2] = RGB[3 * id + 1] = RGB[3 * id] = (unsigned char)(l * 255);
  } else {
    float v1, v2;
    float hue = (float)h / 360;

    v2 = (l < 0.5) ? (l * (1 + s)) : ((l + s) - (l * s));
    v1 = 2 * l - v2;

    RGB[3 * id + 2] = (unsigned char)(255 * HueToRGB(v1, v2, hue + (1.0f / 3)));
    RGB[3 * id + 1] = (unsigned char)(255 * HueToRGB(v1, v2, hue));
    RGB[3 * id] = (unsigned char)(255 * HueToRGB(v1, v2, hue - (1.0f / 3)));
  }
}
float HueToRGB(float v1, float v2, float vH) {
  if (vH < 0)
    vH += 1;

  if (vH > 1)
    vH -= 1;

  if ((6 * vH) < 1)
    return (v1 + (v2 - v1) * 6 * vH);

  if ((2 * vH) < 1)
    return v2;

  if ((3 * vH) < 2)
    return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

  return v1;
}
