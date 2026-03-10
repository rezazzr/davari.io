export interface Education {
  name: string;
  name2?: string;
  link: string;
  link2?: string;
  date: string;
  descr: string;
  logoFile: string;
  logoFile2?: string;
}

export const education: Education[] = [
  {
    name: "Mila",
    name2: "Concordia University",
    link: "https://mila.quebec/en/",
    link2: "https://www.concordia.ca/",
    date: "Jan 2021 - May 2025",
    descr: `<br> <ul>
          <li>PhD of Computer Science: Continual Learning (GPA: 4.0)</li>
          <li>Supervisors: <a href="http://eugenium.github.io/" target="_blank">Dr. Eugene Belilovsky</a></li>
          <li>Scholarships:
              <ul>
                <li> Concordia Merit Scholarship (2021) </li>
                <li> Graduate Doctoral Incentive Fellowship (2021) </li>
              </ul>
          </li>
        </ul>`,
    logoFile: "mila.png",
    logoFile2: "concordia.png",
  },
  {
    name: "Concordia University",
    link: "https://www.concordia.ca/",
    date: "Sep 2017 - May 2020",
    descr: `<br> <ul>
          <li>Master of Computer Science: Natural Language Processing (GPA: 3.95)</li>
          <li>Supervisors: <a href="https://users.encs.concordia.ca/~kosseim/" target="_blank">Dr. Leila Kosseim</a> and <a href="https://users.encs.concordia.ca/~bui/" target="_blank">Dr. Tien Bui</a> </li>
          <li>Scholarships:
              <ul>
                <li> Concordia Merit Scholarship (2018 and 2019) </li>
                <li> Concordia University Conference and Exposition Award (2019) </li>
                <li> IVADO Student Grant (2020) </li>
              </ul>
          </li>
        </ul>`,
    logoFile: "concordia.png",
  },
  {
    name: "McGill University",
    link: "https://www.mcgill.ca/",
    date: "Sep 2013 - Dec 2016",
    descr: `<br> <ul> <li>Bachelor of Science: Double Major in Mathematics and Computer Science</li> <li>Represented McGill University in Putnam Competition in both 2014 and 2015</li> </ul>`,
    logoFile: "mcgill.png",
  },
  {
    name: "Dawson College",
    link: "https://www.dawsoncollege.qc.ca/",
    date: "Sep 2011 - May 2013",
    descr: `<br> <ul> <li>DEC in Pure and Applied Science</li> <li>Dawson Mathematics Competition: 1<sup>st</sup> place in 2012 and 2<sup>nd</sup> place in 2011 </li> </ul>`,
    logoFile: "dawson.png",
  },
];
