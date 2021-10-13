


import imageio




def main() -> None:
	""" """

	img1 = imageio.imread('1.JPG')
	img1_ = img1
	img1 = rgb2gray(img1)

	img2 = imageio.imread('2.JPG')
	img2_ = img2
	img2 = rgb2gray(img2)

	img3 = imageio.imread('3.JPG')
	img3_ = img3
	img3 = rgb2gray(img3)

	points1 = detectHarrisFeatures(img1)
	points2 = detectHarrisFeatures(img2)
	points3 = detectHarrisFeatures(img3)

	f1, vpts1 = extractFeatures(img1, points1)
	f2, vpts2 = extractFeatures(img2, points2)
	f3, vpts3 = extractFeatures(img3, points3)

	pairs12 = matchFeatures(f1, f2)
	pairs23 = matchFeatures(f2, f3)

	matchedPoints12_1 = vpts1[pairs12[:, 1]]
	matchedPoints12_2 = vpts2[pairs12[:, 2]]
	matchedPoints23_2 = vpts2[pairs23[:, 1]]
	matchedPoints23_3 = vpts3[pairs23[:, 2]]

	matchedTriplets = []

	for i=1:size(pairs12,1),
	    match = find(pairs23(:,1) == pairs12(i,2));
	    if(size(match,1)~= 0)
	        matchedTriplets = [matchedTriplets; pairs12(i,1), pairs12(i,2), pairs23(match, 2)];
	    end;
	end;

	Tri, matchper, m, m1 = RANSACTrifocal(vpts1, vpts2, vpts3, matchedTriplets)

	for i in range(6):
	    m(i) = MLEupdate(Tri, m(i))

	shapeInserter = vision.ShapeInserter('Shape', 'Rectangles', 'BorderColor', 'Custom', 'CustomBorderColor', uint8([255 120 0]), 'Fill', true, 'FillColor', 'Custom', 'CustomFillColor', uint8([255, 0, 0]));
	figure(1) 
	RGB1 = img1_;
	hold on;
	for i in range(6):
	    circle1 = int16([m(i).a1(1) m(i).a1(2) 20 20]);
	    RGB1 = step(shapeInserter, RGB1, circle1);
	    # RGB1 = insertMarker(RGB1,int16(m(i).a1(1:2)'),'x', 'color', 'white', 'size', 10);

	plt.imshow(RGB1)
	figure(2)

	RGB2 = img2_
	for i=1:6,
	    circle2 = int16([m(i).a2(1) m(i).a2(2) 20 20]);
	    RGB2 = step(shapeInserter, RGB2, circle2);
	    # RGB2 = insertMarker(RGB2,int16(m(i).a2(1:2)'),'x', 'color', 'white', 'size', 10);

	plt.imshow(RGB2)
	figure(3)
	hold on
	RGB3 = img3_;
	for i in range(6):
	    circle3 = int16([m(i).a3(1) m(i).a3(2) 20 20]);
	    RGB3 = step(shapeInserter, RGB3, circle3);
	    #RGB3 = insertMarker(img3_,int16(m(i).a3(1:2)'),'x', 'color', 'white', 'size', 10);

	plt.imshow(RGB3)

	for i=1:6,
	    p3 = pointTransfer(Tri, m1(i).a1, m1(i).a2)
	    p3 = p3/p3(3)
	    m1(i).a3 = p3

	plt.figure()
	shapeInserter = vision.ShapeInserter('Shape', 'Rectangles', 'BorderColor', 'Custom', 'CustomBorderColor', uint8([0 120 255]), 'Fill', true, 'FillColor', 'Custom', 'CustomFillColor', uint8([0, 0, 255]));
	for i in range(6):
	    circle1 = int16([m1(i).a1(1) m1(i).a1(2) 20 20]);
	    RGB1 = step(shapeInserter, RGB1, circle1);
	    #RGB1 = insertMarker(RGB1,int16(m(i).a1(1:2)'),'x', 'color', 'white', 'size', 10);

	plt.imshow(RGB1);
	figure(2)
	hold on;

	for i in range(6):
	    circle2 = int16([m1(i).a2(1) m1(i).a2(2) 20 20])
	    RGB2 = step(shapeInserter, RGB2, circle2)
	    #RGB2 = insertMarker(RGB2,int16(m(i).a2(1:2)'),'x', 'color', 'white', 'size', 10);
	
	plt.imshow(RGB2);
	figure(3)
	hold on;
	for i in range(6):
	    circle3 = int16([m1(i).a3(1) m1(i).a3(2) 20 20]);
	    RGB3 = step(shapeInserter, RGB3, circle3);
	    #RGB3 = insertMarker(img3_,int16(m(i).a3(1:2)'),'x', 'color', 'white', 'size', 10);
	
	plt.imshow(RGB3)


	plt.figure(3)
	imshow(step(markerInserter,img1,[int16(m(1).a3(1)), int16(m(1).a3(2))]) );
	%plot(m(1).a1(3), m(1).a2(3), 'Marker','p','Color',[.88 .48 0],'MarkerSize',20);


	%%
	%figure(1);
	%showMatchedFeatures(img1, img2, matchedPoints12_1, matchedPoints12_2);

	%figure(2);
	%showMatchedFeatures(img2, img3, matchedPoints23_2, matchedPoints23_3);



