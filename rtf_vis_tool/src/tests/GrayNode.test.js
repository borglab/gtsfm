/* Unit Tests for the GrayNode.js Component.

Author: Adi Singh
*/
import React from 'react';

// Third-Party Package Imports.
import Enzyme, { shallow } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

// Local Imports.
import GrayNode from '../Components/GrayNode';

// Configure enzyme for current React version.
Enzyme.configure({ adapter: new Adapter() })

describe('GrayNode.js Test', () => {

    /*  Creates a sample BlueNode.js component and checks whether it renders with gray background
        and black text as the default.
    */
    it('Node has gray background and black text.', () => {
        const sampleNodeInfo = {
            text: 'sample-text',
            topOffset: 10,
            leftOffset: 20
        }

        const wrapper = shallow(<GrayNode nodeInfo={sampleNodeInfo}/>);
        const div = wrapper.find('GtsfmNode');
        expect(div.props().backgroundColor).toEqual('#dfe8e6');
        expect(div.props().textColor).toEqual('black');
    })
})