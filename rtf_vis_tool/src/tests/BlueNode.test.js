/* Unit Test for the BlueNode.js Component.

Author: Adi Singh
*/
import React from 'react';

// Third-Party Package Imports.
import Enzyme, { shallow } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

// Local Imports.
import BlueNode from '../Components/BlueNode';

// Configure enzyme for current React version.
Enzyme.configure({ adapter: new Adapter() })

describe('BlueNode.js Test', () => {

    /*  Creates a sample BlueNode.js component and checks whether it renders with blue background
        and white text as the default.
    */
    it('Node has blue background and white text.', () => {
        const sampleNodeInfo = {
            text: 'sample-text',
            topOffset: 10,
            leftOffset: 20
        }

        const wrapper = shallow(<BlueNode nodeInfo={sampleNodeInfo}/>);
        const div = wrapper.find('GtsfmNode');
        expect(div.props().backgroundColor).toEqual('#2255e0');
        expect(div.props().textColor).toEqual('white');
    })
})