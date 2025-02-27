struct Dims {
  float xmin;
  float xmax;
  float ymin;
  float ymax;
  int width;
  int height;
};
__kernel void mandlebrot(__global uchar *RGB,__global struct Dims *dims)
{ 
  int id = get_global_id(0);
  int w = dims->width;
  int h = dims->height;
  float fid = (float)id;
  int w2 = w*w;
  float col = (float)(id %w);
  float p_x = (col)/((float)h) * (dims->xmax - dims->xmin ) +  (dims->xmin)  ;
  float p_y = ((float)fid) /((float)w2)*(dims->ymax - dims->ymin ) +  (dims->ymin)  ;
  float x = 0.0;
  float y = 0.0;
  float next_x = 0.0;
  float next_y = 0.0;
  int i = 0;

  for(; i < 256; i += 1){
    next_x = x*x - y*y + p_x;
    next_y = x*y + x*y + p_y;
    if(next_x*next_x + next_y*next_y > 4) break;
    x = next_x;
    y = next_y;
  }
  float val = (float)(i & 0xff);
  RGB[3*id] = sqrt(val/256)*256;
  RGB[3*id + 1] = 0x00;//(char)((id >> 8) % 0xff);
  RGB[3*id + 2] = 0x00;//(char)(id % 0xff);

}
