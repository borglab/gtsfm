/* Unit Test for the PointSizeSlider.js Component.

Author: Adi Singh
*/
import React from 'react';

// Third-Party Package Imports.
import Enzyme, { shallow } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

// Local Imports.
import PointSizeSlider from '../Components/PointSizeSlider';

// Configure enzyme for current React version.
Enzyme.configure({ adapter: new Adapter() })

describe('PointSizeSlider.js Test', () => {

    /*  Checks if the main text of PointSizeSlider.js Component renders properly.
    */
    it('PointSizeSlider.js Text Renders Properly.', () => {
        const wrapper = shallow(<PointSizeSlider />);
        const pTag = wrapper.find('p');
        expect(pTag.first().text()).toEqual('Adjust Point Radius:');
    });
})