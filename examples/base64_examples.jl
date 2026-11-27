## Base64 Encoding and Decoding Examples
using Base64

## string
# Example string to encode and decode
original_string = "Hello, World!"

# Encode the string to Base64
encoded_string = base64encode(original_string)
println("Encoded: ", encoded_string)

# Decode the Base64 string back to the original string
decoded_string = base64decode(encoded_string)
println("Decoded: ",String(decoded_string))

## Vector{UInt8}
# Example vector of bytes to encode and decode
original_bytes = UInt8[72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33]

# Encode the vector to Base64
encoded_bytes = base64encode(original_bytes)
println("Encoded: ", encoded_bytes)

# Decode the Base64 string back to the original vector
decoded_bytes = base64decode(encoded_bytes)
println("Decoded: ", String(decoded_bytes))

## File I/O
img_path = joinpath(@__DIR__, ".", "202010_rufus.jpg")
out_path = joinpath(@__DIR__, ".", "restored_rufus.png")
image_b64 = open(base64encode, img_path)

write(out_path, base64decode(image_b64))