# Overlay Helper Library, used by application or the plugins

Overlay helper library is a set of class-es to implement an abstraction and also support multiple image format overlay. The readme is primarily to list down steps to include new overlay functions.

## How to enable new overlay function?

To enable, any new overlay or rendering function from the graphics pipeline context.

NOTE: We support only opencv based implementation only for now

* Define the overlay function. like this
For ex:

`
bool Yuv420::draw_polygon (std::map<std::string, unsigned char *> & buf_ptr_m, std::string & buffer_name);

`
Plese note the return type should be bool and should return true or false.

* Include/Register the function with a rendering stage name.

For ex: If you wish to register a function for bounding box stage rendering in the graphics pipeline, In the
class constructor

`
register_fn("bbox", &simaai::overlay::Yuv420::draw_rectangle);

`
This would register the draw_rectangle with the name "bbox", this is called from the stage processing of the application.


# TODO

* Create factory like interface?
* Use of image base class
* More cv2 overlay functions
* Optimize `get_nv12_image`
* More instrumentation
* Use of SIMD
