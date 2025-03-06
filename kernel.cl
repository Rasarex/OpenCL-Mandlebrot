struct Dims {
  double xmin;
  double xmax;
  double ymin;
  double ymax;
  int width;
  int height;
  double zoom;
  int zoom_level;
};
void HSLToRGB(double h, double s, double l, __global uchar *RGB, int id);
inline double pow1_5(double x){
  return pown(x,1)/sqrt(x);
}
double HueToRGB(double v1, double v2, double vH);
__kernel void mandlebrot(__global uchar *RGB, __global struct Dims *dims) {
  int id = get_global_id(0);
  int w = dims->width;
  int h = dims->height;
  double fid = (double)id;
  int w2 = w * w;
  double col = (double)(id % w);
  int max_iter = 1000;
  int sub_sample = 1;
    double view_width = (dims->xmax - dims->xmin);
    double view_height = (dims->ymax - dims->ymin);
    double p_x = (col) / ((double)h) * view_width + (dims->xmin);
    double p_y = ((double)fid) / ((double)w2)* view_height + (dims->ymin);
    double x = 0.0;
    double y = 0.0;
    double next_x = 0.0;
    double next_y = 0.0;
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
    double val = (double)(i % max_iter);
    if (i < max_iter){
      double log_zn = log(sqrt(x*x + y*y));
      double nu =  log2(log_zn / log((double)max_iter));
      val = i - nu;
    }
  // little endian // BGR
  double val_hue = (double)(val / (double)max_iter);
  double offset = 220.0;
  double range = 220.0;
  range = fmod(range,360.0);
  double hue = val_hue == 0.0 ? 0.0 :  fmod((pow1_5(val_hue ) * range ) + offset, 360.0);
  HSLToRGB(hue, 1.0,  sqrt(val/(double)max_iter), RGB, id);
}
void HSLToRGB(double h, double s, double l, __global uchar *RGB, int id) {
  unsigned char r = 0;
  unsigned char g = 0;
  unsigned char b = 0;

  if (s == 0) {
    RGB[3 * id + 2] = RGB[3 * id + 1] = RGB[3 * id] = (unsigned char)(l * 255);
  } else {
    double v1, v2;
    double hue = (double)h / 360;

    v2 = (l < 0.5) ? (l * (1 + s)) : ((l + s) - (l * s));
    v1 = 2 * l - v2;

    RGB[3 * id ] = (unsigned char)(255 * HueToRGB(v1, v2, hue + (1.0f / 3)));
    RGB[3 * id + 1] = (unsigned char)(255 * HueToRGB(v1, v2, hue));
    RGB[3 * id + 2] = (unsigned char)(255 * HueToRGB(v1, v2, hue - (1.0f / 3)));
  }
}
double HueToRGB(double v1, double v2, double vH) {
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
// vim: filetype=c
