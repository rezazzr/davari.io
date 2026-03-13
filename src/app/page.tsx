import Image from "next/image";
import { siteConfig } from "@/data/site-config";
import { careers } from "@/data/careers";
import { education } from "@/data/education";
import { teachingExp } from "@/data/teaching-exp";
import SkillsRadarChart from "@/components/SkillsRadarChart";
import Timeline from "@/components/Timeline";
import ParticleNetwork from "@/components/ParticleNetwork";
import RevealOnScroll from "@/components/RevealOnScroll";

export default function HomePage() {
  return (
    <div className="space-y-12">
      <div className="flex flex-col gap-8 lg:flex-row">
        <div className="flex flex-col items-center lg:w-7/12">
          <div className="relative inline-block">
            <ParticleNetwork />
            <Image
              src="/assets/img/reza_profile.png"
              alt={siteConfig.owner.name}
              width={320}
              height={320}
              className="rounded-2xl"
              priority
            />
          </div>
          <RevealOnScroll className="mt-6">
            <h2 className="text-xl font-bold">About</h2>
            <p className="mt-2 leading-relaxed text-text-muted">
              I am a Senior Applied Scientist at{" "}
              <a href="https://www.microsoft.com" target="_blank" className="text-primary hover:underline">Microsoft</a>,
              working on the Office Word team 🤖 My focus is on building agentic
              systems that are better at personalization, context management, and
              tool use. I also research multimodal evaluation methods and design
              benchmarks for agentic flows, drawing on continual learning
              techniques to keep improving the system.
            </p>
            <p className="mt-3 leading-relaxed text-text-muted">
              Before Microsoft, I did my PhD at{" "}
              <a href="https://mila.quebec" target="_blank" className="text-primary hover:underline">Mila</a> and{" "}
              <a href="https://www.concordia.ca" target="_blank" className="text-primary hover:underline">Concordia University</a>,
              where I studied continual, federated, and self-supervised learning
              for NLP and Computer Vision 📚 under the supervision of{" "}
              <a href="http://eugenium.github.io/" target="_blank" className="text-primary hover:underline">Dr. Eugene Belilovsky</a>.
            </p>
            <p className="mt-3 leading-relaxed text-text-muted">
              Outside of work, I am an avid sourdough baker 🍞 If you want
              tips, need help debugging your bake, or are in the Redmond, WA
              area and would like a starter, feel free to{" "}
              <a href={`mailto:${siteConfig.email}`} className="text-primary hover:underline">reach out</a> 😊
            </p>
          </RevealOnScroll>
        </div>

        <RevealOnScroll className="lg:w-5/12" delay={200}>
          <SkillsRadarChart />
        </RevealOnScroll>
      </div>

      <div className="flex flex-col gap-12 lg:flex-row">
        <div className="lg:w-7/12">
          <Timeline animated title="Career" items={careers} />
        </div>

        <div className="space-y-12 lg:w-5/12">
          <Timeline animated title="Education" items={education} />
          <Timeline animated title="Teaching Experience" items={teachingExp} />
        </div>
      </div>
    </div>
  );
}
